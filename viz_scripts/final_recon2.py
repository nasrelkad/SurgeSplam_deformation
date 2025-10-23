import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
from time import time

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette, transform_to_frame, apply_svf, SVFDeformer
from utils.slam_external import build_rotation

w2cs = []

# ---------------------- small utilities ----------------------

def _to_cuda_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device='cuda', dtype=dtype)
    arr = torch.as_tensor(x, dtype=dtype)
    return arr.to('cuda')


def _maybe_rebuild_svf(params: dict, device: str = 'cuda') -> dict:
    """Rebuild an SVFDeformer from saved svf_L* tensors in params.npz.
    Expects levels shaped [1,3,D,D,D]. Copies weights and restores AABB.
    Sets params['svf'] callable and ensures params['svf_n_squarings'] exists.
    """
    # already callable? done
    if 'svf' in params and callable(params['svf']):
        return params

    # drop non-callable placeholders
    if 'svf' in params and not callable(params['svf']):
        params.pop('svf')

    level_keys = sorted([k for k in params.keys() if isinstance(k, str) and k.startswith('svf_L')],
                        key=lambda s: int(s.replace('svf_L', '')) if s.replace('svf_L', '').isdigit() else 1e9)
    if not level_keys:
        return params  # nothing to rebuild

    # infer per-level grid sizes from [1,3,D,D,D]
    Ds = []
    for k in level_keys:
        L = params[k]
        if not isinstance(L, torch.Tensor):
            L = torch.as_tensor(L)
            params[k] = L
        if L.ndim != 5 or L.shape[1] != 3:
            raise ValueError(f"{k} has shape {tuple(L.shape)}, expected [1,3,D,D,D]")
        Ds.append(int(L.shape[2]))
    levels = tuple(Ds)

    svf = SVFDeformer(levels=levels).to(device)
    with torch.no_grad():
        for i, k in enumerate(level_keys):
            getattr(svf, f"svf_L{i}").copy_(params[k].to(device=device, dtype=torch.float32))

        if 'svf_aabb_center' in params and 'svf_aabb_half' in params:
            svf.register_buffer('aabb_center', torch.as_tensor(params['svf_aabb_center'], device=device, dtype=torch.float32), persistent=True)
            svf.register_buffer('aabb_half',   torch.as_tensor(params['svf_aabb_half'],   device=device, dtype=torch.float32), persistent=True)

    params['svf'] = svf
    params['svf_n_squarings'] = int(params.get('svf_n_squarings', 8))
    print(f"[SVF] Rebuilt with levels={levels}, n_squarings={params['svf_n_squarings']}")
    return params


def _aa_to_quat(aa: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle vectors to quaternions."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)
    half = 0.5 * theta
    k = torch.sin(half) / theta
    xyz = aa * k
    w = torch.cos(half)
    return torch.cat([w, xyz], dim=-1)


def _qmul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication with layout [w,x,y,z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dim=-1)


def _select_time_component(value, time_idx: int):
    """Extract per-frame tensor for 'simple' deformation storage formats."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Encountered empty sequence while selecting time component.")
        idx = min(time_idx, len(value) - 1)
        return _to_cuda_tensor(value[idx])

    if isinstance(value, torch.Tensor):
        v = value
        if v.dim() >= 3 and v.shape[-1] > time_idx:
            return v[..., time_idx]
        if v.dim() >= 3 and v.shape[0] > time_idx and v.shape[1] in (1, 3, 4):
            return v[time_idx]
        if v.dim() == 2 and v.shape[0] > time_idx and v.shape[1] in (1, 3, 4):
            return v[time_idx]
        return v

    return _select_time_component(_to_cuda_tensor(value), time_idx)


# ---------------------- deformation wrapper ----------------------

def deform_gaussians(params: dict, time_idx: int, deform_type: str = 'svf'):
    """Return time-deformed gaussian attributes depending on stored deformation model."""
    deform_type = (deform_type or 'svf').lower()
    params['_deform_type'] = deform_type

    def _fallback():
        means = params['means3D']
        if not isinstance(means, torch.Tensor):
            means = _to_cuda_tensor(means)
        return (
            means,
            _to_cuda_tensor(params['unnorm_rotations']),
            _to_cuda_tensor(params['log_scales']),
            _to_cuda_tensor(params['logit_opacities']),
            _to_cuda_tensor(params['rgb_colors']),
        )

    if deform_type == 'svf' and ('svf' in params) and callable(params['svf']):
        local_means = apply_svf(params, time_idx, n_squarings=int(params['svf_n_squarings']))
        return (
            local_means,
            params['unnorm_rotations'],
            params['log_scales'],
            params['logit_opacities'],
            params['rgb_colors'],
        )

    if deform_type == 'gaussian' and all(k in params for k in ['deform_weights', 'deform_stds', 'deform_biases']):
        weights = params['deform_weights']
        stds = params['deform_stds']
        biases = params['deform_biases']
        if not isinstance(weights, torch.Tensor):
            return _fallback()
        time_val = torch.tensor(float(time_idx), device=weights.device, dtype=weights.dtype)
        time_diff = torch.abs(time_val - biases)
        N = min(5, biases.shape[-1])
        _, top_indices = torch.topk(-time_diff, N, dim=1)
        mask = torch.zeros_like(time_diff, dtype=weights.dtype)
        mask.scatter_(1, top_indices, 1.0)

        masked_weights = weights * mask
        masked_biases = biases * mask
        gaussian_term = torch.exp(-0.5 * ((time_val - masked_biases) / (stds + 1e-8)) ** 2)
        deform = torch.sum(masked_weights * gaussian_term, dim=1)

        deform_xyz = deform[:, :3]
        deform_rots = deform[:, 3:7]
        deform_scales = deform[:, 7:10]

        xyz = params['means3D'] + deform_xyz
        rots = params['unnorm_rotations'] + deform_rots
        scales = params['log_scales'] + deform_scales
        opacities = params['logit_opacities']
        colors = params['rgb_colors']
        return xyz, rots, scales, opacities, colors

    if deform_type == 'cv' and all(k in params for k in ['cv_vel_xyz', 'cv_vel_log_scales', 'cv_angvel_aa']):
        means = _to_cuda_tensor(params['means3D'])
        dtype = means.dtype
        device = means.device
        t = torch.tensor(float(time_idx), device=device, dtype=dtype)
        xyz = means + params['cv_vel_xyz'] * t
        base_q = F.normalize(params['unnorm_rotations'], dim=-1)
        dq = _aa_to_quat(params['cv_angvel_aa'] * t)
        rots = F.normalize(_qmul(base_q, dq), dim=-1)
        scales = params['log_scales'] + params['cv_vel_log_scales'] * t
        return xyz, rots, scales, params['logit_opacities'], params['rgb_colors']

    if deform_type == 'simple':
        means = _select_time_component(params['means3D'], time_idx)
        rots = _select_time_component(params['unnorm_rotations'], time_idx)
        scales = _select_time_component(params['log_scales'], time_idx)
        opacities = _select_time_component(params['logit_opacities'], time_idx)
        colors = _select_time_component(params['rgb_colors'], time_idx)
        return (
            _to_cuda_tensor(means),
            _to_cuda_tensor(rots),
            _to_cuda_tensor(scales),
            _to_cuda_tensor(opacities),
            _to_cuda_tensor(colors),
        )

    return _fallback()


# ---------------------- camera / scene loading ----------------------

def load_camera(cfg: dict, scene_path: str):
    raw = dict(np.load(scene_path, allow_pickle=True))
    org_w = raw['org_width']; org_h = raw['org_height']
    w2c = raw['w2c']
    intr = raw['intrinsics']
    K = intr[:3, :3].copy()
    # scale to viz size
    K[0, :] *= cfg['viz_w'] / org_w
    K[1, :] *= cfg['viz_h'] / org_h
    return w2c, K


def _load_params_cuda(scene_path: str) -> dict:
    raw = dict(np.load(scene_path, allow_pickle=True))
    params = {}
    # move everything we care about to CUDA
    for k, v in raw.items():
        try:
            params[k] = _to_cuda_tensor(v)
        except Exception:
            # ragged lists (e.g., per-frame lists)
            try:
                params[k] = [_to_cuda_tensor(v[i]) for i in range(len(v))]
            except Exception:
                params[k] = v  # keep as-is
    return params


def load_scene_data(scene_path: str, first_frame_w2c, K_np, time_idx: int, deform_type: str = 'svf'):
    print(f"Loading data from {scene_path}")
    params = _load_params_cuda(scene_path)

    # prepare per-frame cameras from tracked cam_unnorm_rots / cam_trans
    all_w2cs = []
    if ('cam_unnorm_rots' in params) and ('cam_trans' in params):
        T = params['cam_unnorm_rots'].shape[-1]
        for t in range(T):
            q = F.normalize(params['cam_unnorm_rots'][..., t])
            R = build_rotation(q)
            w2c_t = torch.eye(4, device='cuda', dtype=torch.float32)
            w2c_t[:3, :3] = R
            w2c_t[:3, 3]  = params['cam_trans'][..., t]
            all_w2cs.append(w2c_t.detach().cpu().numpy())
    else:
        all_w2cs.append(np.asarray(first_frame_w2c))

    # rebuild SVF if present in the .npz
    params = _maybe_rebuild_svf(params, device='cuda')
    loaded_type = deform_type
    if 'deform_type' in params:
        raw = params['deform_type']
        if isinstance(raw, (str, bytes)):
            loaded_type = str(raw)
        elif isinstance(raw, np.ndarray):
            try:
                loaded_type = str(raw.item())
            except Exception:
                loaded_type = deform_type
    deform_type = (loaded_type or 'svf')

    # initial deformation at given time
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(params, time_idx, deform_type=deform_type)

    # camera-space points for depth/silhouette path expect identity W2C
    pts_cam = transform_to_frame(local_means, params, time_idx, gaussians_grad=False, camera_grad=False)

    rendervar = {
        'means3D': local_means,
        'rotations': F.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(local_means, device='cuda'),
    }
    if 'feature_rest' in params and params['feature_rest'] is not None:
        rendervar['scales'] = torch.exp(local_scales)
        rendervar['shs'] = torch.cat((
            local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
            params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)
        ), dim=1)
    elif local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        rendervar['colors_precomp'] = local_colors
    else:
        rendervar['scales'] = local_scales
        rendervar['colors_precomp'] = local_colors

    depth_rendervar = {
        'means3D': pts_cam,  # camera-space
        'colors_precomp': get_depth_and_silhouette(pts_cam, torch.eye(4, device='cuda', dtype=torch.float32)),
        'rotations': F.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': torch.zeros_like(local_means, device='cuda'),
    }

    return rendervar, depth_rendervar, all_w2cs, params, K_np


# ---------------------- per-frame path ----------------------

def deform_for_frame(params: dict, time_idx: int):
    deform_type = params.get('_deform_type', 'svf')
    # WORLD → deformed
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(params, time_idx, deform_type=deform_type)
    # WORLD → CAMERA for depth/silhouette
    pts_cam = transform_to_frame(local_means, params, time_idx, gaussians_grad=False, camera_grad=False)

    scene_data = {
        'means3D': local_means,  # world coords (renderer uses view/proj)
        'rotations': F.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(local_means, device='cuda'),
    }
    if 'feature_rest' in params and params['feature_rest'] is not None:
        scene_data['scales'] = torch.exp(local_scales)
        scene_data['shs'] = torch.cat((
            local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
            params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)
        ), dim=1)
    elif local_scales.shape[1] == 1:
        scene_data['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        scene_data['colors_precomp'] = local_colors
    else:
        scene_data['scales'] = local_scales
        scene_data['colors_precomp'] = local_colors

    depth_scene = {
        'means3D': pts_cam,  # camera-space
        'colors_precomp': get_depth_and_silhouette(pts_cam, torch.eye(4, device='cuda', dtype=torch.float32)),
        'rotations': scene_data['rotations'],
        'opacities': scene_data['opacities'],
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': scene_data['means2D'],
    }
    return scene_data, depth_scene


# ---------------------- rendering ----------------------

def render(w2c, K, scene_data, scene_depth_data, cfg):
    assert hasattr(K, 'shape') and tuple(K.shape) == (3, 3), f"K must be 3x3, got {type(K)} {getattr(K, 'shape', None)}"
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], K, w2c, cfg['viz_near'], cfg['viz_far'], use_simplification=cfg['gaussian_simplification'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda'),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered,
        )
        im, _, depth = Renderer(raster_settings=white_bg_cam)(**scene_data)
        return im, depth, _


def rgbd2pcd(color, depth, w2c, K, cfg):
    width, height = color.shape[2], color.shape[1]
    CX, CY = K[0][2], K[1][2]
    FX, FY = K[0][0], K[1][1]

    # pixel grid
    xx = torch.tile(torch.arange(width, device='cuda'), (height,))
    yy = torch.repeat_interleave(torch.arange(height, device='cuda'), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * z, yy * z, z), dim=-1)
    ones = torch.ones(height * width, 1, device='cuda', dtype=torch.float32)
    pts4 = torch.cat((pts_cam, ones), dim=1)
    c2w = torch.inverse(torch.as_tensor(w2c, device='cuda', dtype=torch.float32))
    pts = (c2w @ pts4.T).T[:, :3]

    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    if cfg['render_mode'] == 'depth':
        cols = z
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


# ---------------------- main visualization ----------------------

def visualize(scene_path, cfg, experiment):
    time_idx = 0
    w2c_0, K_np = load_camera(cfg, scene_path)
    deform_type = experiment.config.get('deforms', {}).get('deform_type', 'svf')

    # initial load (also rebuilds SVF)
    scene_data, scene_depth_data, all_w2cs, params, K_np = load_scene_data(scene_path, w2c_0, K_np, time_idx, deform_type=deform_type)
    deform_type = params.get('_deform_type', deform_type)

    # persist intrinsics for the loop
    base_K = torch.as_tensor(K_np, device='cuda', dtype=torch.float32)

    # create viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']), visible=True)

    # first frame render to seed point cloud
    im, depth, _ = render(w2c_0, base_K, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c_0, base_K, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    # optional camera frustums
    if cfg.get('visualize_cams', False):
        w = cfg['viz_w']; h = cfg['viz_h']
        frustum_size = 0.4
        num_t = len(all_w2cs)
        cam_centers = []
        cmap = plt.get_cmap('cool'); norm_factor = 0.5
        for i_t in range(num_t):
            fr = o3d.geometry.LineSet.create_camera_visualization(w, h, K_np, all_w2cs[i_t], frustum_size)
            fr.paint_uniform_color(np.array(cmap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(fr)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
        # trajectory lines
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        for lt in range(total_num_lines):
            cols.append(np.array(cmap((lt * norm_factor / total_num_lines) + norm_factor)[:3]))
        cols = np.array(cols)
        lineset = o3d.geometry.LineSet()
        pts_arr = np.array(cam_centers)
        lineset.points = o3d.utility.Vector3dVector(pts_arr)
        line_indices = np.stack((np.arange(1, len(pts_arr)), np.arange(0, len(pts_arr)-1)), -1)
        lineset.lines = o3d.utility.Vector2iVector(line_indices.astype(np.int32))
        lineset.colors = o3d.utility.Vector3dVector(cols)
        vis.add_geometry(lineset)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    T = max(1, len(all_w2cs))

    # initialize view control to mimic final_recon behaviour
    view_control = vis.get_view_control()
    cam_params = o3d.camera.PinholeCameraParameters()
    view_w2c = np.asarray(w2c_0).copy()
    if cfg.get('offset_first_viz_cam', False):
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5], dtype=view_w2c.dtype)
    cam_params.extrinsic = view_w2c
    view_k = (np.asarray(K_np) * cfg['view_scale']).copy()
    view_k[2, 2] = 1.0
    cam_params.intrinsic.intrinsic_matrix = view_k
    cam_params.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    cam_params.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    ts = time()

    while True:
        # advance time and choose the sequence camera (NOT the viewer camera)
        time_idx = (time_idx + 1) % T
        scene_data, scene_depth_data = deform_for_frame(params, time_idx)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix.copy()
        render_K = (view_k / cfg['view_scale']).astype(np.float32)
        render_K[2, 2] = 1.0
        w2c_view = cam_params.extrinsic.copy()

        if time() - ts > 0.033:
            w2cs.append(w2c_view.copy())
            if len(w2cs) == (613 // 8) * 7 + 7:
                np.save("w2cs.npy", np.array(w2cs))
                vis.destroy_window()
                return
            print(len(w2cs))
            ts = time()

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data.get('colors_precomp', torch.zeros_like(params['means3D'])).contiguous().double().cpu().numpy())
        else:
            im, depth, _ = render(w2c_view, render_K, scene_data, scene_depth_data, cfg)
            pts, cols = rgbd2pcd(im, depth, w2c_view, render_K, cfg)

        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(experiment.config["workdir"], experiment.config["run_name"]) 
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]

    viz_cfg = experiment.config["viz"]
    visualize(scene_path, viz_cfg, experiment)
