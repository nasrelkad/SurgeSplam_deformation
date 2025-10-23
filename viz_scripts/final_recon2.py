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
from utils.slam_helpers import get_depth_and_silhouette, transform_to_frame, maybe_rebuild_mlp, apply_mlp_deformation
from utils.slam_external import build_rotation

w2cs = []

# ---------------------- small utilities ----------------------

def _to_cuda_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device='cuda', dtype=dtype)
    arr = torch.as_tensor(x, dtype=dtype)
    return arr.to('cuda')
# ---------------------- deformation wrapper ----------------------

def deform_gaussians(params: dict, time_idx: int):
    """Return time-deformed gaussian attributes using the saved MLP deformation."""
    if 'mlp_deformer' not in params:
        raise RuntimeError("This viewer expects params saved with an MLP deformer.")
    local_means = apply_mlp_deformation(params, time_idx, allow_gaussian_grad=False)
    return (
        local_means,
        params['unnorm_rotations'],
        params['log_scales'],
        params['logit_opacities'],
        params['rgb_colors'],
    )


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


def load_scene_data(scene_path: str, first_frame_w2c, K_np, time_idx: int):
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

    params = maybe_rebuild_mlp(params, device='cuda')
    params['_deform_type'] = 'mlp'
    if 'mlp_deformer' not in params:
        raise RuntimeError("params.npz does not contain an MLP deformation field.")

    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(params, time_idx)

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
    if 'mlp_deformer' not in params:
        raise RuntimeError("MLP deformer missing in params; this viewer only supports the MLP variant.")
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(params, time_idx)
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
    scene_data, scene_depth_data, all_w2cs, params, K_np = load_scene_data(scene_path, w2c_0, K_np, time_idx)

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
