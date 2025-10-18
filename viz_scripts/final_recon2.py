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
from utils.slam_helpers import get_depth_and_silhouette, _graph_node_motion
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
                        
def deform_gaussians(params, time, deform_grad, N=5, deformation_type='gaussian', temperature=0.5):
    """
    Gaussian deformation:
      - soft attention over basis functions (keeps gradients flowing)
      - safe positive stds via softplus
      - rotation update via quaternion multiplication (delta quat)
      - scale updated additively in log domain
    
    """
    if deformation_type == 'gaussian':
        # Pull tensors with/without grad
        if deform_grad:
            W = params['deform_weights']      # [G,B,10]
            S_raw = params['deform_stds']     # [G,B,10]
            C = params['deform_biases']       # [G,B,10]
        else:
            W = params['deform_weights'].detach()
            S_raw = params['deform_stds'].detach()
            C = params['deform_biases'].detach()

        # Make sure 'time' is a tensor on the right device/dtype and broadcastable to C
        t = torch.as_tensor(time, device=C.device, dtype=C.dtype)
        while t.dim() < C.dim():
            t = t.unsqueeze(0)  # expand to match [G,B,10] by broadcasting

        # strictly positive bandwidths
        S = F.softplus(S_raw) + 1e-3  # [G,B,10]

        # attention-like weights across bases (dim=1)
        # score = - (t - C)^2 / (2*S^2)
        score = -((t - C) ** 2) / (2.0 * (S ** 2) + 1e-12)
        attn = F.softmax(score / max(temperature, 1e-3), dim=1)  # sum_B(attn)=1

        # Optional soft top-k: keep top-N mass but renormalize (still differentiable)
        if N is not None and isinstance(N, int) and 0 < N < attn.shape[1]:
            topv, topi = torch.topk(attn, k=N, dim=1)
            m = torch.zeros_like(attn)
            m.scatter_(1, topi, 1.0)
            attn = attn * m
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted sum over bases -> per-gaussian deformation vector (10)
        deform = torch.sum(attn * W, dim=1)  # [G,10]

        # Split into xyz(3), rot_quat_delta(4), dlog_scales(3)
        deform_xyz    = deform[:, 0:3]
        deform_rot_q  = deform[:, 3:7]   # interpret as delta quaternion
        deform_dlogS  = deform[:, 7:10]

        # Apply updates
        # positions
        xyz = params['means3D'] + deform_xyz

        # rotations: compose base quaternion with delta quaternion
        base_q = F.normalize(params['unnorm_rotations'], dim=-1)      # [G,4]
        dq = F.normalize(deform_rot_q, dim=-1)                         # [G,4]

        # quaternion multiply: (base_q * dq)
        w1, x1, y1, z1 = base_q.unbind(-1)
        w2, x2, y2, z2 = dq.unbind(-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        rots = torch.stack([w, x, y, z], dim=-1)
        rots = F.normalize(rots, dim=-1)

        # scales: additive in log-domain (stays positive after exp in renderer)
        scales = params['log_scales'] + deform_dlogS

        opacities = params['logit_opacities']
        colors = params['rgb_colors']
        return xyz, rots, scales, opacities, colors

    elif deformation_type == 'graph':
        """
        Skin each Gaussian to K nearest graph nodes, transform by node motion (CA + optional Fourier),
        then blend in Lie/log spaces.
        """
        
        # dt in frames relative to 0 (if you keep absolute time indices)
        dt = (time if torch.is_tensor(time) else torch.tensor(float(time), device=params['means3D'].device, dtype=params['means3D'].dtype)).float()
        # node motions
        use_fourier = ('fourier_xyz_cos' in params)
        R_i, t_i, logs_i = _graph_node_motion(params, dt, use_fourier=use_fourier)   # [Gm,3,3], [Gm,3], [Gm,3]

        x = params['means3D']                      # [G,3]

        
        device = x.device
        idx = params['graph_idx'].to(dtype=torch.long, device=device)   # must be long on same device
        nodes = params['graph_nodes'].to(device=device)
        w = params['graph_w'].to(device=device)

        # now safe to index
        n_ik    = nodes[idx]           # [G,K,3]
        t_ik    = t_i[idx]             # [G,K,3]
        R_ik    = R_i[idx]             # [G,K,3,3]
        logs_ik = logs_i[idx]          # [G,K,3]

        # center-relative, rotate, then uncenter and translate:
        # x'_k = R_i(t)*(x - n_i) + n_i + t_i(t)
        x_rel = (x.unsqueeze(1) - n_ik)                         # [G,K,3]
        x_def = (R_ik @ x_rel.unsqueeze(-1)).squeeze(-1) + n_ik + t_ik
        x_out = (w.unsqueeze(-1) * x_def).sum(dim=1)            # [G,3]

        # rotation: blend axis-angle from node rotations (small-angle OK). We reuse node ang_i used inside _graph_node_motion
        # For stability, compute ang_i again to avoid storing; tiny overhead.
        # If you prefer, store the ang_i in _graph_node_motion and return it.
        # Here: approximate Gaussian rotation as identity (or keep original) — optional:
        rots_out = params['unnorm_rotations']  # keep original per-Gaussian orientation; advanced: blend node rotvecs.

        # scales: blend log-scales
        logs_g = (w.unsqueeze(-1) * logs_ik).sum(dim=1)         # [G,3]
        scales_out = params['log_scales'] + logs_g              # add on top of per-Gaussian log-scales (optional)

        opacities = params['logit_opacities']
        colors    = params['rgb_colors']
        return x_out, rots_out, scales_out, opacities, colors

    elif deformation_type == 'simple':

        xyz = params['means3D']
        rots = params['unnorm_rotations']
        scales = params['log_scales']
        opacities = params['logit_opacities']
        colors = params['rgb_colors']
        return xyz, rots, scales, opacities, colors

    elif deformation_type == 'cv':  # constant velocity (add 'ca' with t^2 terms if desired)
        t = torch.as_tensor(time, device=params['means3D'].device, dtype=params['means3D'].dtype).view(-1,1)
        v_xyz  = params['cv_vel_xyz']          # [G,3]
        v_lsg  = params['cv_vel_log_scales']   # [G,3]
        w_aa   = params['cv_angvel_aa']        # [G,3]

        xyz = params['means3D'] + v_xyz * t
        base_q = F.normalize(params['unnorm_rotations'], dim=-1)
        dq = _aa_to_quat(w_aa * t)
        rots = F.normalize(_qmul(base_q, dq), dim=-1)
        scales = params['log_scales'] + v_lsg * t

        opacities = params['logit_opacities']; colors = params['rgb_colors']
        return xyz, rots, scales, opacities, colors

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


def load_scene_data(scene_path, first_frame_w2c, intrinsics, time_idx,deformation_type = None):
    # Load Scene Data
    print(f"Loading data from {scene_path}")
    raw = dict(np.load(scene_path, allow_pickle=True))
    

    # Only keep fields needed for rendering
    KEEP = {
        'means3D', 'rgb_colors', 'log_scales', 'logit_opacities',
        'unnorm_rotations', 'cam_unnorm_rots', 'cam_trans',
        'intrinsics', 'w2c', 'timestep'
    }

    params = {}
    for k, v in raw.items():
        if k not in KEEP:
            continue  # skip graph_conf, debug blobs, etc.
        tval = _to_torch_cuda(v)
        if tval is None:    # non-numeric (dict/None/str) → skip
            continue
        params[k] = tval

    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device='cuda')
    first_frame_w2c = torch.as_tensor(first_frame_w2c, dtype=torch.float32, device='cuda')

    # Convert everything best-effort to tensors
    params = {}
    for k, v in raw.items():
        params[k] = _to_torch_cuda(v)

    def _Nx3(x):
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, 3, device='cuda', dtype=torch.float32)
            # stack [3] vectors
            return torch.stack([e.view(-1) for e in x], dim=0).contiguous().view(-1,3)
        x = x.to(dtype=torch.float32, device='cuda')
        if x.ndim == 1:
            if x.numel() % 3 != 0:
                raise ValueError(f"_Nx3: 1D size {x.numel()} not multiple of 3")
            x = x.view(-1, 3)
        elif x.ndim == 2 and x.size(-1) == 3:
            pass
        elif x.ndim == 2 and x.size(0) == 3:
            x = x.t()
        else:
            raise ValueError(f"_Nx3: unexpected shape {tuple(x.shape)}")
        return x.contiguous()

    def _Nx1(x):
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, 1, device='cuda', dtype=torch.float32)
            x = torch.stack([e.view(-1) for e in x], dim=0)
        x = x.to(dtype=torch.float32, device='cuda')
        if x.ndim == 1:
            x = x.view(-1, 1)
        elif x.ndim == 2 and x.size(-1) == 1:
            pass
        else:
            x = x.view(-1, 1)
        return x.contiguous()

    # Coerce fields if present
    if 'means3D' in params:          params['means3D'] = _Nx3(params['means3D'])
    if 'rgb_colors' in params:       params['rgb_colors'] = _Nx3(params['rgb_colors']).clamp_(0, 1)
    if 'log_scales' in params:       params['log_scales'] = _Nx3(params['log_scales'])
    if 'logit_opacities' in params:  params['logit_opacities'] = _Nx1(params['logit_opacities'])
    if 'unnorm_rotations' in params: params['unnorm_rotations'] = (
        torch.stack(params['unnorm_rotations'], dim=0).float().contiguous()
        if isinstance(params['unnorm_rotations'], list) else params['unnorm_rotations'].float().contiguous()
    )

    # Intrinsics / w2c (optional)
    if 'intrinsics' in params: params['intrinsics'] = params['intrinsics'].float().contiguous()
    if 'w2c' in params:        params['w2c'] = params['w2c'].float().contiguous()

    # Camera stacks (cam_unnorm_rots [4, T], cam_trans [3, T]) can arrive as lists or objects — coerce:
    for key in ['cam_unnorm_rots', 'cam_trans']:
        if key in params:
            val = params[key]
            if isinstance(val, list):
                # stack list elements on last dim if each is [4] or [3]
                elems = [e if isinstance(e, torch.Tensor) else torch.as_tensor(e, device='cuda', dtype=torch.float32) for e in val]
                if elems[0].ndim == 1:
                    params[key] = torch.stack(elems, dim=-1).contiguous()
                else:
                    params[key] = torch.stack(elems, dim=-1).contiguous()
            else:
                params[key] = val.to(device='cuda', dtype=torch.float32)

    # Prepare all w2c per frame
    all_w2cs = []
    if 'cam_unnorm_rots' in params and 'cam_trans' in params:
        rot = params['cam_unnorm_rots']  # [4,T] or list
        tran = params['cam_trans']       # [3,T] or list
        if isinstance(rot, list):  rot = torch.stack(rot, dim=-1)
        if isinstance(tran, list): tran = torch.stack(tran, dim=-1)
        rot = rot.to(dtype=torch.float32, device='cuda')
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
        # fallback: use provided first frame
        all_w2cs.append(first_frame_w2c.cpu().numpy())

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())
        


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
        'colors_precomp': get_depth_and_silhouette(transformed_pts, first_frame_w2c),
        'rotations': torch.nn.functional.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        'scales': torch.exp(torch.tile(local_scales, (1, 3))),
        'means2D': torch.zeros_like(local_means, device="cuda")
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