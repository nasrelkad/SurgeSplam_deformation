#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)
from utils.recon_helpers import setup_camera # you already have these
from utils.slam_helpers import (
    deform_gaussians as deform_gaussians_eval,
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    xyzt_time_gate,
    apply_xyzt_gate,
    rehydrate_endo4dgs,
)

"""
Updated final_recon.py

- dynamic playback over all timesteps
- uses SAME deformation as SLAM
- uses XYZT gate (dynamic) but softened for nicer video
- saves frames to disk so you can ffmpeg them into a video

Usage:
    python final_recon.py --scene path/to/params.npz --out dynamic_out --dynamic
    ffmpeg -r 30 -i dynamic_out/frame_%04d.png -c:v libx264 -pix_fmt yuv420p colon_dynamic.mp4
"""


def load_params(scene_path, device="cuda"):
    data = np.load(scene_path, allow_pickle=True)
    params = {}
    net_state = data['endo4dgs_net_state'] if 'endo4dgs_net_state' in data.files else None
    net_meta = data['endo4dgs_net_meta'] if 'endo4dgs_net_meta' in data.files else None
    for k in data.files:
        if k in ('endo4dgs_net_state', 'endo4dgs_net_meta'):
            continue
        arr = data[k]
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr).to(device)
            # train code saves some scalars as 0-d arrays
            if t.dtype == torch.float64:
                t = t.float()
            params[k] = t
        else:
            params[k] = arr
    if net_state is not None:
        params = rehydrate_endo4dgs(
            params,
            state=net_state,
            meta=net_meta,
            device=device
        )
    return params


def build_w2cs(params):
    """
    Rebuild per-frame world-to-camera matrices from saved cam_unnorm_rots + cam_trans
    """
    cam_rots = F.normalize(params["cam_unnorm_rots"], dim=1)  # [3, T] or [B,4,T] in some setups
    cam_trans = params["cam_trans"]  # [3, T]
    T = cam_trans.shape[-1]
    w2cs = []
    from utils.slam_external import build_rotation  # you have this

    for t in range(T):
        rot = build_rotation(cam_rots[..., t])
        trans = cam_trans[..., t]
        w2c = torch.eye(4, device=rot.device, dtype=rot.dtype)
        w2c[:3, :3] = rot
        w2c[:3, 3] = trans
        w2cs.append(w2c)
    return w2cs


def make_intrinsics_from_params(params, viz_w, viz_h):
    """
    Your runs usually save 'k' or separate fx,fy,cx,cy.
    Try to reconstruct a 3x3.
    """
    if "k" in params:
        k = params["k"].clone()
        k[0, 2] = min(k[0, 2], viz_w - 1)
        k[1, 2] = min(k[1, 2], viz_h - 1)
        return k
    else:
        fx = params.get("fx", torch.tensor(800., device="cuda"))
        fy = params.get("fy", torch.tensor(800., device="cuda"))
        cx = params.get("cx", torch.tensor(viz_w / 2., device="cuda"))
        cy = params.get("cy", torch.tensor(viz_h / 2., device="cuda"))
        k = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], device="cuda").float()
        return k


def render_dynamic(params, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)

    # figure out how many frames we have
    if "num_frames" in params:
        num_frames = int(params["num_frames"].item())
    else:
        # fall back to camera count
        num_frames = params["cam_trans"].shape[-1]

    w2cs = build_w2cs(params)
    device = params["means3D"].device

    viz_w = cfg.get("viz_w", 640)
    viz_h = cfg.get("viz_h", 480)
    k = make_intrinsics_from_params(params, viz_w, viz_h)

    for t in tqdm(range(num_frames), desc="render dynamic"):
        # 1) deform to time t (uses SAME code as SLAM)
        local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians_eval(
            params,
            float(t),
            False,                      # no grad
            N=cfg.get("N_bases_eval", 5),
            deformation_type=cfg.get("deformation_type", "gaussian"),
        )

        # 2) build renderer input
        rendervar = transformed_params2rendervar(
            params,
            local_means,
            local_rots,
            local_scales,
            local_opacities,
            local_colors,
        )

        # 3) temporal gate (SOFT)
        if "t_mu" in params and "t_logvar" in params:
            time_gate = xyzt_time_gate(params, float(t))  # [G]
            # soften so gaussians don't blink
            time_gate = 0.3 * time_gate + 0.7
            rendervar = apply_xyzt_gate(rendervar, time_gate, gate_thresh=0.0)
            if "opacities" in rendervar:
                rendervar["opacities"] = torch.clamp(rendervar["opacities"], min=0.05)

        # 4) setup camera
        w2c = w2cs[t]
        cam = setup_camera(
            viz_w,
            viz_h,
            k,
            w2c,
            cfg.get("viz_near", 0.01),
            cfg.get("viz_far", 20.0),
            use_simplification=cfg.get("gaussian_simplification", False),
        )

        # 5) render
        im, depth_sil, _ = Renderer(
            w2c,
            8,              # sh degree – same as training
            rendervar,
            {},             # depth data not needed for viz
            cfg,
            cam=cam,
        )

        # 6) save rgb
        im_np = (im.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        import imageio
        imageio.imwrite(out_path, im_np)


def render_single(params, out_path, cfg, time_idx=None):
    if time_idx is None:
        if "num_frames" in params:
            time_idx = int(params["num_frames"].item()) // 2
        else:
            time_idx = params["cam_trans"].shape[-1] // 2

    w2cs = build_w2cs(params)
    device = params["means3D"].device

    viz_w = cfg.get("viz_w", 640)
    viz_h = cfg.get("viz_h", 480)
    k = make_intrinsics_from_params(params, viz_w, viz_h)

    # deform
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians_eval(
        params,
        float(time_idx),
        False,
        N=cfg.get("N_bases_eval", 5),
        deformation_type=cfg.get("deformation_type", "gaussian"),
    )

    rendervar = transformed_params2rendervar(
        params,
        local_means,
        local_rots,
        local_scales,
        local_opacities,
        local_colors,
    )

    if "t_mu" in params and "t_logvar" in params:
        time_gate = xyzt_time_gate(params, float(time_idx))
        time_gate = 0.3 * time_gate + 0.7
        rendervar = apply_xyzt_gate(rendervar, time_gate, gate_thresh=0.0)
        if "opacities" in rendervar:
            rendervar["opacities"] = torch.clamp(rendervar["opacities"], min=0.05)

    w2c = w2cs[time_idx]
    cam = setup_camera(
        viz_w,
        viz_h,
        k,
        w2c,
        cfg.get("viz_near", 0.01),
        cfg.get("viz_far", 20.0),
        use_simplification=cfg.get("gaussian_simplification", False),
    )

    im, depth_sil, _ = Renderer(
        w2c,
        8,
        rendervar,
        {},
        cfg,
        cam=cam,
    )

    im_np = (im.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    import imageio
    imageio.imwrite(out_path, im_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="path to params.npz")
    parser.add_argument("--out", default="dynamic_out", help="output dir / image")
    parser.add_argument("--dynamic", action="store_true", help="export all frames")
    parser.add_argument("--time", type=int, default=None, help="single frame index")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = load_params(args.scene, device=device)

    # viz cfg – can be tweaked
    cfg = dict(
        viz_w=640,
        viz_h=480,
        viz_near=0.01,
        viz_far=20.0,
        gaussian_simplification=False,
        deformation_type="gaussian",
        N_bases_eval=5,
    )

    if args.dynamic:
        render_dynamic(params, args.out, cfg)
    else:
        # if out is a dir, write frame.png inside it
        if os.path.isdir(args.out):
            os.makedirs(args.out, exist_ok=True)
            out_path = os.path.join(args.out, "frame.png")
        else:
            out_path = args.out
        render_single(params, out_path, cfg, time_idx=args.time)


if __name__ == "__main__":
    main()
