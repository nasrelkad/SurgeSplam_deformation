import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation

from time import time
w2cs = []

def deform_gaussians(params, time, deform_grad, N=5, deformation_type='gaussian'):
    """
    Calculate deformations using the N closest basis functions based on |time - bias|.
    Returns updated xyz, rots, scales, opacities, colors.
    """
    if deformation_type == 'gaussian':
        if deform_grad:
            weights = params['deform_weights']
            stds = params['deform_stds']
            biases = params['deform_biases']
        else:
            weights = params['deform_weights'].detach()
            stds = params['deform_stds'].detach()
            biases = params['deform_biases'].detach()

        time_diff = torch.abs(time - biases)
        _, top_indices = torch.topk(-time_diff, N, dim=1)  # smallest diffs
        mask = torch.zeros_like(time_diff, dtype=torch.float)
        mask.scatter_(1, top_indices, 1.0)

        masked_weights = weights * mask
        masked_biases = biases * mask

        deform = torch.sum(
            masked_weights * torch.exp(-1 / (2 * stds**2) * (time - masked_biases)**2), dim=1
        )  # Nx10

        deform_xyz = deform[:, :3]
        deform_rots = deform[:, 3:7]
        deform_scales = deform[:, 7:10]

        xyz = params['means3D'] + deform_xyz
        rots = params['unnorm_rotations'] + deform_rots
        scales = params['log_scales'] + deform_scales
        opacities = params['logit_opacities']
        colors = params['rgb_colors']

    elif deformation_type == 'simple':
        with torch.no_grad():
            xyz = params['means3D'][time]
            rots = params['unnorm_rotations'][time]
            scales = params['log_scales'][time]
            opacities = params['logit_opacities'][time]
            colors = params['rgb_colors'][time]
    else:
        raise ValueError(f"Unknown deformation_type: {deformation_type}")

    return xyz, rots, scales, opacities, colors


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3].copy()

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics, time_idx, deformation_type=None):
    print(f"Loading data from {scene_path}")
    all_params = dict(np.load(scene_path, allow_pickle=True))
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()
    try:
        all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}

        keys = [k for k in all_params.keys() if
                k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                          'gt_w2c_all_frames', 'cam_unnorm_rots',
                          'cam_trans', 'keyframe_time_indices']]

        params = all_params
        for k in keys:
            if not isinstance(all_params[k], torch.Tensor):
                params[k] = torch.tensor(all_params[k]).cuda().float()
            else:
                params[k] = all_params[k].cuda().float()
    except:
        params = {}
        for key in all_params.keys():
            try:
                params[key] = torch.tensor(all_params[key]).cuda()
            except:
                params[key] = [torch.tensor(all_params[key][i]).cuda()
                               for i in range(all_params[key].shape[0])]

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(
        params, time_idx, False, deformation_type=deformation_type
    )
    transformed_pts = local_means

    rendervar = {
        'means3D': transformed_pts,
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(local_means, device="cuda")
    }
    if "feature_rest" in params:
        rendervar['scales'] = torch.exp(local_scales)
        rendervar['shs'] = torch.cat(
            (
                local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
                params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)
            ),
            dim=1
        )
    elif local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        rendervar['colors_precomp'] = local_colors
    else:
        rendervar['scales'] = local_scales
        rendervar['colors_precomp'] = local_colors

    depth_rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, first_frame_w2c),
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': torch.zeros_like(local_means, device="cuda")
    }
    return rendervar, depth_rendervar, all_w2cs, params


def deform_and_render(params, time_idx, deformation_type, w2c):
    w2c = torch.tensor(w2c).cuda().float()
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(
        params, time_idx, False, deformation_type=deformation_type
    )
    transformed_pts = local_means

    rendervar = {
        'means3D': transformed_pts,
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(local_means, device="cuda")
    }
    if "feature_rest" in params:
        rendervar['scales'] = torch.exp(local_scales)
        rendervar['shs'] = torch.cat(
            (
                local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
                params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)
            ),
            dim=1
        )
    elif local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        rendervar['colors_precomp'] = local_colors
    else:
        rendervar['scales'] = local_scales
        rendervar['colors_precomp'] = local_colors

    depth_rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': torch.zeros_like(local_means, device="cuda")
    }
    return rendervar, depth_rendervar, params


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, nl in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - nl), -1)[nl:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'],
                           use_simplification=cfg['gaussian_simplification'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        return im, depth, _


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([0, 0, 0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(scene_path, cfg, experiment):
    # Load Scene Data
    time_idx = 0
    deformation_type = experiment.config['deforms']['deform_type']
    w2c, k = load_camera(cfg, scene_path)
    scene_data, scene_depth_data, all_w2cs, params = load_scene_data(
        scene_path, w2c, k, time_idx, deformation_type=deformation_type
    )

    # Open3D window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, _ = render(w2c, k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Visualize cameras if requested
    if cfg['visualize_cams']:
        frustum_size = 0.4
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(0, num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines) + norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

    # Initialize view control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c.copy()
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # ---- Video writers for Gaussian rasterizer outputs ----
    out_w = int(cfg['viz_w'])
    out_h = int(cfg['viz_h'])
    fps = int(cfg.get('fps', 30))
    write_depth_video = bool(cfg.get('save_depth_video', True))  # toggle via config

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter("gauss_render.mp4", fourcc, fps, (out_w, out_h))
    depth_writer = cv2.VideoWriter("gauss_depth.mp4", fourcc, fps, (out_w, out_h)) if write_depth_video else None

    ts = time()
    # Interactive Rendering
    while True:
        scene_data, scene_depth_data, params = deform_and_render(
            params, time_idx, deformation_type, all_w2cs[time_idx]
        )

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if time() - ts > 1.0 / fps:
            # Control how many poses to store; keep original logic
            if len(w2cs) == 613 // 8 * 7 + 7:
                np.save("w2cs.npy", np.array(w2cs))
                # do not exit, allow final video frames to flush
                pass
            else:
                w2cs.append(w2c)
            ts = time()

        # Always render Gaussian rasterizer outputs
        im, depth, _ = render(w2c, k, scene_data, scene_depth_data, cfg)  # im: (3,H,W), depth: (1,H,W)

        # Update Open3D visualization (optional, but preserved)
        pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        # ---- Write RGB frame (rasterizer) ----
        rgb = (im.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # (H,W,3) RGB
        if (rgb.shape[1], rgb.shape[0]) != (out_w, out_h):
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # ---- Write depth frame (colored) if enabled ----
        if depth_writer is not None:
            d = depth[0].detach().cpu().numpy()  # (H,W)
            d_valid = d[np.isfinite(d) & (d > 0)]
            if d_valid.size > 0:
                d_min = np.percentile(d_valid, 1.0)
                d_max = np.percentile(d_valid, 99.0)
                if d_max <= d_min:
                    d_max = d_min + 1e-6
                d_norm = np.clip((d - d_min) / (d_max - d_min), 0, 1)
            else:
                d_norm = np.zeros_like(d)
            d_u8 = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)  # BGR
            if (d_color.shape[1], d_color.shape[0]) != (out_w, out_h):
                d_color = cv2.resize(d_color, (out_w, out_h), interpolation=cv2.INTER_AREA)
            depth_writer.write(d_color)

        # Timestep advance / exit
        if time_idx >= len(all_w2cs) - 1:
            break
        else:
            time_idx += 1

    # Cleanup
    rgb_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    # Optional toggles for video output (if not already in your config)
    viz_cfg.setdefault("fps", 30)
    viz_cfg.setdefault("save_depth_video", True)

    # Visualize & record
    visualize(scene_path, viz_cfg, experiment)
