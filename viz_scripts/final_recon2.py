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

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette, _graph_node_motion, apply_svf, SVFDeformer  
from utils.slam_external import build_rotation


from time import time
w2cs = []


# def deform_gaussians(params, time, deform_grad, N=5,deformation_type = 'gaussian'):
#     """
#     Calculate deformations using the N closest basis functions based on |time - bias|.

#     Args:
#         params (dict): Dictionary containing deformation parameters.
#         time (torch.Tensor): Current time step.
#         deform_grad (bool): Whether to calculate gradients for deformations.
#         N (int): Number of closest basis functions to consider.

#     Returns:
#         xyz (torch.Tensor): Updated 3D positions.
#         rots (torch.Tensor): Updated rotations.
#         scales (torch.Tensor): Updated scales.
#     """
#     if deformation_type =='gaussian':
#         if deform_grad:
#             weights = params['deform_weights']
#             stds = params['deform_stds']
#             biases = params['deform_biases']
#         else:
#             weights = params['deform_weights'].detach()
#             stds = params['deform_stds'].detach()
#             biases = params['deform_biases'].detach()

#         # Calculate the absolute difference between time and biases
#         time_diff = torch.abs(time - biases)

#         # Get the indices of the N smallest time differences
#         _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

#         # Create a mask to select only the top N basis functions
#         mask = torch.zeros_like(time_diff, dtype=torch.float)
#         mask.scatter_(1, top_indices, 1.0).detach()

#         # Register a gradient hook to zero out gradients for irrelevant basis functions
#         if deform_grad:
#             def zero_out_irrelevant_gradients(grad):
#                 return grad * mask

#             weights.register_hook(zero_out_irrelevant_gradients)
#             biases.register_hook(zero_out_irrelevant_gradients)
#             stds.register_hook(zero_out_irrelevant_gradients)

#         # Calculate deformations
#         deform = torch.sum(
#             weights * torch.exp(-1 / (2 * stds**2) * (time - biases)**2), dim=1
#         )  # Nx10 gaussians deformations

#         deform_xyz = deform[:, :3]
#         deform_rots = deform[:, 3:7]
#         deform_scales = deform[:, 7:10]

#         xyz = params['means3D'] + deform_xyz
#         rots = params['unnorm_rotations'] + deform_rots
#         scales = params['log_scales'] + deform_scales

    
#     elif deformation_type == 'simple':
#         with torch.no_grad():
#             xyz = params['means3D'][...,time]
#             rots = params['unnorm_rotations'][...,time]
#             scales = params['log_scales'][...,time]
#     # print(deformation_type)
#     return xyz, rots, scales




def _to_torch_cuda(x, float_dtype=torch.float32):
    """Convert numpy/py objects (incl. object arrays) to CUDA tensors.
       Returns None or the original object for non-numeric things."""
    if isinstance(x, torch.Tensor):
        t = x
        if not t.is_floating_point():
            t = t.float()
        return t.to(dtype=float_dtype, device='cuda')

    if isinstance(x, np.ndarray):
        # Fast path: numeric arrays
        if x.dtype != np.object_:
            return torch.as_tensor(x, dtype=float_dtype, device='cuda')

        # Object arrays — handle common cases
        if x.shape == ():  # 0-D object array
            obj = x.item()
            return _to_torch_cuda(obj, float_dtype)

        # 1-D/ND object array → list
        lst = x.tolist()

        # If it's a list/tuple: try to stack elementwise if possible
        if isinstance(lst, (list, tuple)):
            if len(lst) == 0:
                return torch.empty(0, device='cuda', dtype=float_dtype)
            coerced = []
            same_shape = True
            first_shape = None
            for e in lst:
                # Try to convert each element to a numeric ndarray
                if isinstance(e, np.ndarray) and e.dtype != np.object_:
                    arr = e
                else:
                    # Allow plain lists/tuples/numbers
                    try:
                        arr = np.asarray(e)
                    except Exception:
                        return None  # give up: non-numeric (dict/None/str)
                    if arr.dtype == np.object_:
                        return None
                coerced.append(arr)
                if first_shape is None:
                    first_shape = arr.shape
                elif arr.shape != first_shape:
                    same_shape = False
            if same_shape:
                arr = np.stack(coerced, axis=0)
                return torch.as_tensor(arr, dtype=float_dtype, device='cuda')
            else:
                # ragged — return list of tensors
                return [torch.as_tensor(a, dtype=float_dtype, device='cuda') for a in coerced]

        # Not list/tuple: could be dict/str/None
        return None

    # Plain python number?
    try:
        return torch.as_tensor(x, dtype=float_dtype, device='cuda')
    except Exception:
        return None  # non-numeric (dict/str/None)

def _maybe_rebuild_svf(params):
    """
    Rebuild params['svf'] from saved svf_L* tensors if possible.
    Falls back silently if reconstruction fails.
    """
    # Already usable?
    if 'svf' in params and callable(params['svf']):
        return

    # Collect level tensors: svf_L0, svf_L1, ...
    level_keys = [k for k in params.keys() if isinstance(k, str) and k.startswith('svf_L')]
    if not level_keys:
        # nothing saved; ensure no bogus 'svf'
        if 'svf' in params and not callable(params['svf']):
            params.pop('svf')
        return

    # sort levels by index
    try:
        level_keys.sort(key=lambda s: int(s.replace('svf_L', '')))
    except Exception:
        level_keys.sort()

    levels = []
    for k in level_keys:
        v = params[k]
        if isinstance(v, torch.Tensor):
            levels.append(v)
    if not levels:
        return

    try:
        from utils.slam_helpers import SVFDeformer
    except Exception as e:
        print(f"[load_scene_data] SVFDeformer not available ({e}); rendering without SVF.")
        return

    svf = None
    # --- Try constructor that accepts tensors directly
    try:
        svf = SVFDeformer(levels)  # e.g., forward expects preinit tensors
    except TypeError:
        # --- Try constructor that accepts integer resolutions, then load weights
        try:
            resolutions = []
            for L in levels:
                # expect shape like (C, D, H, W) or (D, H, W, C) -> normalize to (C,D,H,W)
                shape = tuple(int(x) for x in L.shape)
                if len(shape) == 4:
                    # heuristics: channels last?
                    if shape[-1] in (2,3):    # (D,H,W,2/3)
                        C = shape[-1]; D,H,W = shape[0],shape[1],shape[2]
                        resolutions.append((C,D,H,W))
                    else:                     # (C,D,H,W)
                        resolutions.append(shape)
                else:
                    # unexpected; bail
                    raise ValueError(f"Unexpected SVF level tensor shape {shape}")
            svf = SVFDeformer(resolutions=resolutions)
            # Try to load weights into similarly named params/buffers if available
            for i, L in enumerate(levels):
                for name, p in list(svf.named_parameters()) + list(svf.named_buffers()):
                    if name.endswith(f"{i}") and p.shape == L.shape:
                        with torch.no_grad():
                            p.copy_(L)
                        break
        except Exception as e2:
            print(f"[load_scene_data] Could not reconstruct SVFDeformer: {e2}")
            svf = None

    if svf is not None:
        svf = svf.cuda()
        params['svf'] = svf
        print(f"[load_scene_data] Reconstructed SVFDeformer with {len(levels)} levels.")
    else:
        # ensure no unusable 'svf' hangs around
        if 'svf' in params and not callable(params['svf']):
            params.pop('svf')

def _ensure_shape_Nx3(x):
    """Force tensor/list into [N,3] float32 CUDA."""
    if isinstance(x, list):
        if len(x) == 0:
            return torch.empty(0, 3, device='cuda', dtype=torch.float32)
        x = [e if isinstance(e, torch.Tensor) else torch.as_tensor(e, device='cuda', dtype=torch.float32) for e in x]
        same = all(e.ndim == 1 and e.numel() == 3 for e in x)
        if same:
            return torch.stack(x, dim=0).contiguous()
        x = torch.stack([e.view(-1) for e in x], dim=0)
    if isinstance(x, torch.Tensor):
        x = x.to(device='cuda', dtype=torch.float32)
        if x.ndim == 1:
            if x.numel() % 3 != 0:
                raise ValueError(f"_ensure_shape_Nx3: 1D size {x.numel()} not multiple of 3")
            x = x.view(-1, 3)
        elif x.ndim == 2 and x.size(-1) == 3:
            pass
        elif x.ndim == 2 and x.size(0) == 3:
            x = x.t()
        else:
            raise ValueError(f"_ensure_shape_Nx3: unexpected shape {tuple(x.shape)}")
        return x.contiguous()
    raise TypeError("_ensure_shape_Nx3 expects tensor or list")

def _aa_to_quat(aa, eps=1e-12):
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)
    half = 0.5 * theta
    k = torch.sin(half) / theta
    xyz = aa * k
    w = torch.cos(half)
    return torch.cat([w, xyz], dim=-1)

def _qmul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2], dim=-1)
                        
def deform_gaussians(params, time, return_all=False, deformation_type='svf', **kwargs):
    # read n_squarings from params if present
    n_squarings = 8
    if 'svf_n_squarings' in params:
        try:
            n_squarings = int(params['svf_n_squarings'])
        except Exception:
            n_squarings = int(getattr(params['svf_n_squarings'], 'item', lambda: 8)())

    has_callable_svf = ('svf' in params) and callable(params['svf'])

    if deformation_type in ('svf', 'gaussian') and has_callable_svf:
        local_means = apply_svf(params, time, n_squarings=n_squarings)
    else:
        # Either not using SVF here, or we couldn't reconstruct it from .npz
        local_means = params['means3D']

    local_rots      = params['unnorm_rotations']
    local_scales    = params['log_scales']
    local_opacities = params['logit_opacities']
    local_colors    = params['rgb_colors']

    if return_all:
        return local_means, local_rots, local_scales, local_opacities, local_colors
    return local_means, local_rots, local_scales, local_opacities, local_colors


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics, time_idx, deformation_type=None):
    """
    Load a saved scene for visualization.

    Pipeline:
      SVF(world) -> transform_to_frame(world->cam, time_idx) -> render
      (no second W2C in the depth/silhouette path)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from utils.slam_helpers import (
        build_rotation,
        transform_to_frame,
        get_depth_and_silhouette,
    )

    # ---------- helpers ----------
    def _to_torch_cuda(v):
        if isinstance(v, torch.Tensor):
            return v.to('cuda')
        try:
            t = torch.as_tensor(v)
            # only move numeric-like arrays
            return t.to('cuda') if t.dtype != torch.bool else t.to('cuda')
        except Exception:
            return None

    def _Nx3(x):
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, 3, device='cuda', dtype=torch.float32)
            x = torch.stack([torch.as_tensor(e, device='cuda', dtype=torch.float32).view(-1) for e in x], dim=0)
        else:
            x = x.to(dtype=torch.float32, device='cuda')
        if x.ndim == 1:            x = x.view(-1, 3)
        elif x.ndim == 2:
            if x.shape[-1] == 3:   pass
            elif x.shape[0] == 3:  x = x.t()
            else:                  x = x.reshape(-1, 3)
        else:                      x = x.reshape(-1, 3)
        return x.contiguous()

    def _Nx1(x):
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, 1, device='cuda', dtype=torch.float32)
            x = torch.stack([torch.as_tensor(e, device='cuda', dtype=torch.float32).view(-1) for e in x], dim=0)
        else:
            x = x.to(dtype=torch.float32, device='cuda')
        if x.ndim == 1:            x = x.view(-1, 1)
        elif x.ndim == 2 and x.shape[-1] == 1: pass
        else:                      x = x.view(-1, 1)
        return x.contiguous()

    def _maybe_rebuild_svf(params):
        """
        Rebuild params['svf'] from saved level tensors (svf_L0, svf_L1, ...),
        if the callable SVF module isn't present (e.g., loaded from .npz).
        """
        if 'svf' in params and callable(params['svf']):
            return  # already valid

        # collect level tensors
        level_keys = sorted([k for k in params.keys() if isinstance(k, str) and k.startswith('svf_L')],
                            key=lambda s: int(s.replace('svf_L', '')) if s.replace('svf_L', '').isdigit() else 1e9)
        if not level_keys:
            # nothing to rebuild from
            if 'svf' in params and not callable(params['svf']):
                params.pop('svf')
            return

        levels = [params[k] for k in level_keys if isinstance(params[k], torch.Tensor)]
        if not levels:
            return

        # Try to import the deformer class
        try:
            from utils.slam_helpers import SVFDeformer
        except Exception as e:
            print(f"[load_scene_data] SVFDeformer not available ({e}); rendering without SVF.")
            return

        # Build module (support common ctor variants)
        svf = None
        try:
            svf = SVFDeformer(levels)  # most repos
        except TypeError:
            try:
                svf = SVFDeformer(levels=levels)
            except Exception as e:
                print(f"[load_scene_data] Could not reconstruct SVFDeformer: {e}")
                return

        if svf is not None:
            svf = svf.cuda()
            params['svf'] = svf
            # keep level tensors for reference; no need to delete svf_L*
            print(f"[load_scene_data] Reconstructed SVFDeformer with {len(levels)} levels.")

    # ---------- load .npz ----------
    print(f"Loading data from {scene_path}")
    raw = dict(np.load(scene_path, allow_pickle=True))

    # Build params dict (best-effort tensors on CUDA)
    params = {k: _to_torch_cuda(v) for k, v in raw.items()}

    # coerce common arrays
    if 'means3D'          in params and params['means3D']          is not None: params['means3D']          = _Nx3(params['means3D'])
    if 'rgb_colors'       in params and params['rgb_colors']       is not None: params['rgb_colors']       = _Nx3(params['rgb_colors']).clamp_(0, 1)
    if 'log_scales'       in params and params['log_scales']       is not None: params['log_scales']       = _Nx3(params['log_scales'])
    if 'logit_opacities'  in params and params['logit_opacities']  is not None: params['logit_opacities']  = _Nx1(params['logit_opacities'])
    if 'unnorm_rotations' in params and params['unnorm_rotations'] is not None:
        ur = params['unnorm_rotations']
        if isinstance(ur, list):
            ur = torch.stack([torch.as_tensor(e, device='cuda', dtype=torch.float32) for e in ur], dim=0)
        params['unnorm_rotations'] = ur.float().contiguous()

    # intrinsics / first-frame w2c tensors
    intrinsics      = torch.as_tensor(intrinsics, dtype=torch.float32, device='cuda')
    first_frame_w2c = torch.as_tensor(first_frame_w2c, dtype=torch.float32, device='cuda')

    # If a non-callable 'svf' snuck in (e.g., from npz), drop it BEFORE rebuild
    if 'svf' in params and not callable(params['svf']):
        params.pop('svf')

    # Try to reconstruct the SVF module from saved levels (svf_L*)
    _maybe_rebuild_svf(params)

    # ---------- per-frame w2c stack ----------
    all_w2cs = []
    if ('cam_unnorm_rots' in params and params['cam_unnorm_rots'] is not None
        and 'cam_trans' in params and params['cam_trans'] is not None):
        rot  = params['cam_unnorm_rots']
        tran = params['cam_trans']
        if isinstance(rot, list):  rot  = torch.stack(rot,  dim=-1)
        if isinstance(tran, list): tran = torch.stack(tran, dim=-1)
        rot  = rot.to(dtype=torch.float32, device='cuda')
        tran = tran.to(dtype=torch.float32, device='cuda')
        T = rot.shape[-1]
        for t in range(T):
            q = F.normalize(rot[..., t])
            R = build_rotation(q)
            w2c = torch.eye(4, device='cuda', dtype=torch.float32)
            w2c[:3, :3] = R
            w2c[:3, 3]  = tran[..., t]
            all_w2cs.append(w2c.cpu().numpy())
    else:
        all_w2cs.append(first_frame_w2c.detach().cpu().numpy())

    # ---------- deform (world) -> single world->camera transform ----------
    # deform_gaussians should gracefully skip SVF if params['svf'] is absent
    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(
        params, time_idx, False, deformation_type=deformation_type
    )
    pts_cam = transform_to_frame(local_means, params, time_idx, gaussians_grad=False, camera_grad=False)

    # ---------- build render vars ----------
    rendervar = {
        'means3D': pts_cam,
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(pts_cam, device="cuda"),
    }
    if "feature_rest" in params and params["feature_rest"] is not None:
        rendervar['scales'] = torch.exp(local_scales)
        rendervar['shs'] = torch.cat(
            (
                local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
                params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
            ),
            dim=1,
        )
    elif local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        rendervar['colors_precomp'] = local_colors
    else:
        rendervar['scales'] = local_scales
        rendervar['colors_precomp'] = local_colors

    # depth/sil: points already in camera space -> identity W2C to avoid double-transform
    I = torch.eye(4, device=pts_cam.device, dtype=pts_cam.dtype)
    depth_rendervar = {
        'means3D': pts_cam,
        'colors_precomp': get_depth_and_silhouette(pts_cam, I),
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': torch.zeros_like(pts_cam, device="cuda"),
    }

    return rendervar, depth_rendervar, all_w2cs, params



def deform_and_render(params,time_idx,deformation_type,w2c):
    w2c = torch.tensor(w2c).cuda().float()
    local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,time_idx,False,deformation_type=deformation_type)
    transformed_pts = local_means

    rendervar = {
        'means3D': transformed_pts,
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'means2D': torch.zeros_like(local_means, device="cuda")
    }
    if "feature_rest" in params:
        rendervar['scales'] = torch.exp(local_scales)
        rendervar['shs'] = torch.cat((local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2), params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)), dim=1)
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
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'], use_simplification=cfg['gaussian_simplification'])
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
        # depth_sil, _, _ = Renderer(raster_settings=white_bg_cam)(**timestep_depth_data)
        # differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        # sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, _


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([0, 0, 0]) # np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(scene_path, cfg,experiment):
    # Load Scene Data
    time_idx = 0
    deformation_type = experiment.config['deforms']['deform_type']
    w2c, k = load_camera(cfg, scene_path)
    scene_data, scene_depth_data, all_w2cs,params = load_scene_data(scene_path, w2c, k,time_idx,deformation_type=deformation_type)

    # vis.create_window()
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

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
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
        
        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
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
        
    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
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

    ts = time()
    # Interactive Rendering
    while True:
        # scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, w2c, k,time_idx,deformation_type=deformation_type)
        scene_data,scene_depth_data, params = deform_and_render(params,time_idx,deformation_type,all_w2cs[time_idx])

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic
        if time() - ts > 0.033:
            if len(w2cs) == 613 // 8 * 7 + 7: # 765, 700, 613
                np.save("w2cs.npy", np.array(w2cs))
                exit()
            print(len(w2cs))
            w2cs.append(w2c)
            ts = time()
        
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, _ = render(w2c, k, scene_data, scene_depth_data, cfg)
            # if cfg['show_sil']:
            #     im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
            # time_idx = 0
        vis.update_renderer()
        if time_idx >= len(all_w2cs) - 1:
            time_idx = 0
        else:
            time_idx+=1
        
    # Cleanup
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

    # Visualize Final Reconstruction
    visualize(scene_path, viz_cfg,experiment)