import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import energy_mask
import torchvision
import numpy as np
import cv2
import math
from utils.slam_external import build_rotation
from torch import nn

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# def params2rendervar(params):
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': params['rgb_colors'],
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


def transformed_params2rendervar(params, transformed_pts,local_rots,local_scales,local_opacities,local_colors):
    rendervar = {
        'means3D': transformed_pts,
        'rotations': F.normalize(local_rots, dim=-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    
    if local_scales.shape[1] == 1:
        rendervar['colors_precomp'] = local_colors
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
        # print('using uniform scales')
    else:
        try: 
            rendervar['shs'] = torch.cat((local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2), params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)), dim=1)
            rendervar['scales'] = torch.exp(local_scales)
        except: 
            rendervar['colors_precomp'] = local_colors
            rendervar['scales'] = torch.exp(local_scales)
    return rendervar


# def project_points(points_3d, intrinsics):
#     """
#     Function to project 3D points to image plane.
#     params:
#     points_3d: [num_gaussians, 3]
#     intrinsics: [3, 3]
#     out: [num_gaussians, 2]
#     """
#     points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
#     points_2d = points_2d.transpose(0, 1)
#     points_2d = points_2d / points_2d[:, 2:]
#     points_2d = points_2d[:, :2]
#     return points_2d

# def params2silhouette(params):
#     sil_color = torch.zeros_like(params['rgb_colors'])
#     sil_color[:, 0] = 1.0
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': sil_color,
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


# def transformed_params2silhouette(params, transformed_pts):
#     sil_color = torch.zeros_like(params['rgb_colors'])
#     sil_color[:, 0] = 1.0
#     rendervar = {
#         'means3D': transformed_pts,
#         'colors_precomp': sil_color,
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar

def _farthest_point_sample(xyz, M):
    """
    xyz: [N,3] (cuda float)
    returns: [M] indices of FPS (very small N is fine; this is O(NM))
    """
    
    N = xyz.shape[0]
    inds = torch.empty(M, dtype=torch.long, device=xyz.device)
    d2 = torch.full((N,), float("inf"), device=xyz.device)
    # start from a random point (or the centroid)
    inds[0] = torch.randint(0, N, (1,), device=xyz.device)
    last = xyz[inds[0]][None, :]
    for i in range(1, M):
        d2 = torch.minimum(d2, torch.sum((xyz - last) ** 2, dim=-1))
        inds[i] = torch.argmax(d2)
        last = xyz[inds[i]][None, :]
    return inds

def initialize_graph_deformations(params, num_nodes=256, K=6, sigma_mult=0.5,
                                  use_fourier=True, M=2):
    """
      - params['graph_nodes']      : [Gm,3]
      - params['graph_idx']        : [G,K] (long)  KNN node indices per Gaussian
      - params['graph_w']          : [G,K] weights per Gaussian (softmax of -dist^2/σ^2)
      - Per-node learnables (all torch.nn.Parameter):
          node_v_xyz,node_a_xyz     : [Gm,3]
          node_v_ang,node_a_ang     : [Gm,3]  (axis-angle velocity/acc)
          node_v_s,  node_a_s       : [Gm,3]  (log-scale velocity/acc)
          fourier_{xyz,ang,s}_cos/sin : [Gm,M,3] (optional)
    """
    
    means = params['means3D'].detach()  # [G,3], cuda float
    G = means.shape[0]

    # 1) make nodes by FPS
    with torch.no_grad():
        fps_idx = _farthest_point_sample(means, min(num_nodes, G))
        nodes = means[fps_idx]  # [Gm,3]
    Gm = nodes.shape[0]

    # 2) KNN from each Gaussian to nodes
    with torch.no_grad():
        # squared distances
        # [G, Gm]
        d2 = torch.cdist(means, nodes, p=2.0) ** 2
        knn_d2, knn_idx = torch.topk(d2, k=min(K, Gm), dim=1, largest=False)
        # σ = sigma_mult * median node spacing
        # estimate spacing from nodes
        nn_nodes = torch.topk(torch.cdist(nodes, nodes, p=2.0), k=2, dim=1, largest=False).values[:, 1]
        sigma = sigma_mult * torch.median(nn_nodes).clamp_min(1e-6)
        w = torch.softmax(-knn_d2 / (sigma * sigma), dim=1)  # [G,K]

    params['graph_nodes'] = torch.nn.Parameter(nodes.requires_grad_(False))
    params['graph_idx']   = knn_idx.long()       # not a Parameter (kept fixed)
    params['graph_w']     = w                    # not a Parameter (kept fixed)

    # 3) per-node learnables (CA + small periodic)
    zeros = lambda *s: torch.zeros(*s, device=means.device, dtype=means.dtype)
    params['node_v_xyz'] = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))
    params['node_a_xyz'] = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))
    params['node_v_ang'] = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))
    params['node_a_ang'] = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))
    params['node_v_s']   = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))
    params['node_a_s']   = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))

    params['node_ang0']  = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))  # small bias
    params['node_s0']    = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))  # log-scale bias
    params['node_t0']    = torch.nn.Parameter(zeros(Gm, 3).requires_grad_(True))  # translation bias

    if use_fourier and M > 0:
        params['fourier_xyz_cos'] = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))
        params['fourier_xyz_sin'] = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))
        params['fourier_ang_cos'] = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))
        params['fourier_ang_sin'] = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))
        params['fourier_s_cos']   = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))
        params['fourier_s_sin']   = torch.nn.Parameter(zeros(Gm, M, 3).requires_grad_(True))

    # keep a reference timescale if XYZT is not used
    if 'timestep' not in params:
        params['timestep'] = torch.zeros(G, device=means.device)
    return params

def ensure_binding_buffers(params, K: int | None = None, tag: str = ""):
    """
    Force params['graph_idx'] to Long, params['graph_w'] to Float32 (no grad),
    grow rows to match means3D, and fix column count to K if provided.
    Safe to call anytime. Leaves other tensors unchanged.
    """
    if 'graph_idx' not in params or 'graph_w' not in params or 'means3D' not in params:
        return params

    device = params['means3D'].device
    with torch.no_grad():
        gi = params['graph_idx']
        gw = params['graph_w']

        # strip Parameter wrappers
        if isinstance(gi, nn.Parameter): gi = gi.detach()
        if isinstance(gw, nn.Parameter): gw = gw.detach()

        # coerce dtypes/devices
        if gi.dtype != torch.long:       gi = gi.to(torch.long)
        if gi.device != device:          gi = gi.to(device)
        if gw.dtype != torch.float32:    gw = gw.to(torch.float32)
        if gw.device != device:          gw = gw.to(device)
        gi = gi.contiguous(); gw = gw.contiguous()

        # grow rows to match number of gaussians
        G = int(params['means3D'].shape[0])
        curG, curK = gi.shape
        need = G - curG
        if need > 0:
            padK = curK if K is None else int(K)
            gi = torch.cat([gi, torch.zeros(need, padK, dtype=torch.long, device=device)], dim=0)
            gw = torch.cat([gw, torch.zeros(need, padK, dtype=torch.float32, device=device)], dim=0)

        # fix K (columns) if requested
        if K is not None and gi.shape[1] != int(K):
            newK = int(K)
            new_gi = torch.zeros(G, newK, dtype=torch.long, device=device)
            new_gw = torch.zeros(G, newK, dtype=torch.float32, device=device)
            common = min(gi.shape[1], newK)
            new_gi[:, :common] = gi[:G, :common]
            new_gw[:, :common] = gw[:G, :common]
            gi, gw = new_gi, new_gw

        params['graph_idx'] = gi
        params['graph_w']   = gw

    # optional probe
    # print(f"[ensure_binding_buffers {tag}] gi={params['graph_idx'].dtype}, gw={params['graph_w'].dtype}")
    return params


# removed: rebind_graph*


# removed: rebind_graph*

def periodic_rebind_if_far(params, tau_mult=2.0, K=None, sigma_mult=None, stride_tag=None):
    """
    Cheap coverage check: if a Gaussian is far from its current node barycenter,
    rebind it. tau = tau_mult * median node spacing.
    Optional stride_tag lets you call this every N frames w/o extra state.
    """
    if 'graph_nodes' not in params or 'graph_idx' not in params or 'graph_w' not in params:
        return params

    means = params['means3D'].detach()          # [G,3]
    nodes = params['graph_nodes'].detach()      # [Gm,3]
    gi = params['graph_idx']                    # [G,K]
    gw = params['graph_w']                      # [G,K]

    # barycenter of current bindings
    bary = (nodes[gi] * gw[..., None]).sum(dim=1)   # [G,3]
    d = torch.norm(means - bary, dim=-1)            # [G]

    # node spacing → tau
    nn_nodes = torch.topk(torch.cdist(nodes, nodes, p=2.0), k=2, dim=1, largest=False).values[:, 1]
    sigma = torch.median(nn_nodes).clamp_min(1e-6)
    tau = tau_mult * sigma

    to_fix = (d > tau).nonzero().flatten()
    if to_fix.numel() > 0:
        rebind_graph_subset(params, to_fix, K=K, sigma_mult=sigma_mult)
    return params


def _graph_node_motion(params, dt, use_fourier=True):
    """
    dt: scalar float Tensor (frames from t0)
    returns per-node:
      R_i(t)  [Gm,3,3], t_i(t) [Gm,3], logs_i(t) [Gm,3]
    """
    
   
    Gm = params['node_v_xyz'].shape[0]
    # CA terms
    t_i   = params['node_t0'] + params['node_v_xyz'] * dt + 0.5 * params['node_a_xyz'] * (dt * dt)
    ang_i = params['node_ang0'] + params['node_v_ang'] * dt + 0.5 * params['node_a_ang'] * (dt * dt)
    logs  = params['node_s0']   + params['node_v_s']   * dt + 0.5 * params['node_a_s']   * (dt * dt)

    if use_fourier and 'fourier_xyz_cos' in params:
        M = params['fourier_xyz_cos'].shape[1]
        # simple unit frequency set {1...M} (you can scale by 2π/T if you like)
        k = torch.arange(1, M + 1, device=dt.device, dtype=dt.dtype)
        kd = k * dt  # [M]
        sin_kd = torch.sin(kd)
        cos_kd = torch.cos(kd)
        # broadcast to [Gm,M,3] * [M] -> [Gm,M,3]
        t_i   = t_i   + (params['fourier_xyz_cos'] * cos_kd[None, :, None]).sum(1) + \
                        (params['fourier_xyz_sin'] * sin_kd[None, :, None]).sum(1)
        ang_i = ang_i + (params['fourier_ang_cos'] * cos_kd[None, :, None]).sum(1) + \
                        (params['fourier_ang_sin'] * sin_kd[None, :, None]).sum(1)
        logs  = logs  + (params['fourier_s_cos']   * cos_kd[None, :, None]).sum(1) + \
                        (params['fourier_s_sin']   * sin_kd[None, :, None]).sum(1)

    # exp map: axis-angle -> quaternion -> rotation matrix
    # re-use your build_rotation by converting axis-angle to quat
    aa = ang_i
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = aa / theta
    half = 0.5 * theta
    qw = torch.cos(half)
    qxyz = axis * torch.sin(half)
    q = torch.cat([qw, qxyz], dim=-1)  # [Gm,4]
    R = build_rotation(q)               # [Gm,3,3]
    return R, t_i, logs


def get_depth_and_silhouette(pts_3D, w2c=None):
   

    # Convert to tensor, ensure float dtype
    pts_3D = torch.as_tensor(pts_3D)
    if not pts_3D.dtype.is_floating_point:
        pts_3D = pts_3D.float()

    # Handle empty input early
    if pts_3D.numel() == 0:
        # Return whatever your pipeline expects; if it expects a tensor, return an empty [0] vector
        return torch.empty(0, device=pts_3D.device, dtype=pts_3D.dtype)

    # Ensure 2D [N,3]
    if pts_3D.dim() == 1:
        if pts_3D.numel() % 3 != 0:
            raise ValueError(f"get_depth_and_silhouette: 1D pts_3D length {pts_3D.numel()} "
                             f"is not a multiple of 3")
        pts_3D = pts_3D.view(-1, 3)
    elif pts_3D.dim() == 2 and pts_3D.size(-1) != 3:
        raise ValueError(f"get_depth_and_silhouette: expected last dim=3, got {pts_3D.shape}")

    if w2c is None:
        pts_cam = pts_3D  # already in camera space
    else:
    # Ensure w2c is a proper [4,4] (pad if [3,4] or [3,3])
        w2c = torch.as_tensor(w2c, dtype=pts_3D.dtype, device=pts_3D.device)
        if w2c.shape == (3, 4):
            w2c = torch.cat([w2c, torch.tensor([[0, 0, 0, 1.0]], dtype=w2c.dtype, device=w2c.device)], dim=0)
        elif w2c.shape == (3, 3):
            w2c = torch.cat([w2c, torch.zeros(1, 3, dtype=w2c.dtype, device=w2c.device)], dim=0)
            w2c = torch.cat([w2c, torch.tensor([[0, 0, 0, 1.0]], dtype=w2c.dtype, device=w2c.device)], dim=1)
        elif w2c.shape != (4, 4):
            raise ValueError(f"get_depth_and_silhouette: unexpected w2c shape {w2c.shape}")

        # Homogenize and transform
        ones = torch.ones((pts_3D.shape[0], 1), dtype=pts_3D.dtype, device=pts_3D.device)
        pts4 = torch.cat([pts_3D, ones], dim=-1)      # [N,4]
        pts_cam = (w2c @ pts4.T).T                    # [N,4]


    depth_z = pts_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette


# def params2depthplussilhouette(params, w2c):
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_pts,local_rots,local_scales,local_opacities):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(local_rots, dim=-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    if local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
    else:
        rendervar['scales'] = torch.exp(local_scales)
    return rendervar


def initialize_xyzt(params, num_frames: int, xyzt_init_sigma: float = 5.0, device=None):
    """
    params: dict with at least 'means3D'
    Adds:
      - 't_mu':      [G]   temporal mean in frame index space
      - 't_logvar':  [G]   log( sigma_t^2 )  (sigma in frames)
    """
    device = device or params['means3D'].device
    G = params['means3D'].shape[0]

    # If you have a "birth" frame for each Gaussian, use it here; otherwise mid-sequence.
    t_mu0 = torch.full((G,), float(num_frames - 1) / 2.0, device=device)

    # Initialize with moderately wide temporal support (in frames)
    sigma0 = torch.full((G,), xyzt_init_sigma, device=device)
    t_logvar0 = (2.0 * torch.log(sigma0)).detach()

    # Register as parameters if your param dict stores tensors with requires_grad
    params['t_mu']     = torch.nn.Parameter(t_mu0.requires_grad_(True))
    params['t_logvar'] = torch.nn.Parameter(t_logvar0.requires_grad_(True))
    return params

def apply_xyzt_gate(render_vars: dict, gt: torch.Tensor, gate_thresh: float = 0.0):
    """
    Multiply opacities by temporal gate while preserving renderer's expected shapes.
    - Keep opacities as [G,1], not [G].
    - Hard-cull per-G tensors if gate_thresh > 0.
    """
    if not isinstance(render_vars, dict):
        return render_vars

    if gt.ndim != 1:
        gt = gt.reshape(-1)

    # Move gt to the same device as means3D if possible
    if "means3D" in render_vars and torch.is_tensor(render_vars["means3D"]):
        dev = render_vars["means3D"].device
        if gt.device != dev:
            gt = gt.to(dev)

    G = gt.shape[0]

    # 1) Scale an opacity-like channel, preserving [G,1]
    for k in ("opacity", "opacities", "alphas"):
        if k in render_vars and torch.is_tensor(render_vars[k]):
            op = render_vars[k]
            if op.shape[0] == G:
                # Ensure column vector for single-channel opacities
                if op.ndim == 1:
                    op = op.view(G, 1)
                elif op.ndim == 2 and op.shape[1] != 1:
                    # Multi-channel case: broadcast along channel dim
                    op = op * gt.view(G, 1).to(op.dtype)
                    render_vars[k] = op
                    break
                # Standard single-channel case
                render_vars[k] = op * gt.view(G, 1).to(op.dtype)
                break

    # 2) Optional hard cull of per-G tensors
    if gate_thresh > 0.0:
        mask = gt > gate_thresh
        if mask.sum() == 0:
            m = torch.zeros_like(mask); m[int(torch.argmax(gt))] = True
            mask = m
        keys = list(render_vars.keys())
        for k in keys:
            v = render_vars[k]
            if torch.is_tensor(v) and v.shape[:1] == (G,):
                render_vars[k] = v[mask]

    return render_vars

def xyzt_time_gate(params, t: float):
    """
    Returns g_t in [G] for the given time index t (float allowed).
    g_t = exp( -0.5 * (t - mu)^2 / sigma^2 ), sigma^2 = exp(t_logvar)
    """
    mu = params['t_mu']
    logvar = params['t_logvar']
    var = torch.exp(logvar).clamp_min(1e-8)
    diff = t - mu
    gt = torch.exp(-0.5 * (diff * diff) / var)
    return gt

def transform_to_frame(local_means,params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = local_means
    else:
        pts = local_means.detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts


def transform_to_frame_eval(params, local_means, camrt=None, rel_w2c=None):
    """
    Robustly transform Gaussian centers from world to camera frame for evaluation.
    Accepts local_means shaped [N,3], [3], [N,3,1], or [...,3] and normalizes to [N,3].
    """
    # Build or use provided pose
    if rel_w2c is None:
        cam_rot, cam_tran = camrt
        rel_w2c = torch.eye(4, device='cuda', dtype=torch.float32)
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

    # --- Normalize point tensor to shape [N, 3] ---
    pts = local_means
    with torch.no_grad():
        # move last dim to 3 if needed (e.g., [N,3,1] or [*,3])
        if pts.dim() >= 2 and pts.shape[-1] == 1 and pts.shape[-2] == 3:
            pts = pts.squeeze(-1)                  # [N,3,1] -> [N,3]
        if pts.dim() >= 2 and pts.shape[-1] != 3 and pts.shape[-2] == 3:
            pts = pts.transpose(-1, -2)            # [...,3,*] -> [...,*,3]

        # collapse any leading dims to [N,3]
        if pts.dim() > 2:
            pts = pts.reshape(-1, pts.shape[-1])
        if pts.dim() == 1:
            # [3] -> [1,3]
            if pts.numel() == 3:
                pts = pts.unsqueeze(0)
            else:
                raise ValueError(f"local_means has unsupported 1D shape {list(pts.shape)}; expected 3 elements.")

        if pts.shape[-1] != 3:
            raise ValueError(f"local_means must have last dim 3, got shape {list(pts.shape)}")

    pts = pts.detach()

    # Homogeneous transform
    pts_ones = torch.ones(pts.shape[0], 1, device='cuda', dtype=torch.float32)
    pts4 = torch.cat((pts, pts_ones), dim=1)       # [N,4]
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]  # [N,3]

    return transformed_pts


# def mask_params(params, mask):
#     params['means3D'] = params['means3D'][mask]
#     params['rgb_colors'] = params['rgb_colors'][mask]
#     params['unnorm_rotations'] = params['unnorm_rotations'][mask]
#     params['logit_opacities'] = params['logit_opacities'][mask]
#     params['log_scales'] = params['log_scales'][mask]    
#     return params



def transformed_GRNparams2rendervar(params, transformed_pts,local_rots,local_scales,local_opacities,local_colors):
    rendervar = {
        'means3D': transformed_pts,
        'rotations': F.normalize(local_rots, dim =-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    if local_scales.shape[1] == 1:
        rendervar['colors_precomp'] = local_colors
        rendervar['scales'] = torch.tile(local_scales, (1, 3))
        # print('using uniform scales')
    else:
        try: 
            rendervar['shs'] = torch.cat((local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2), params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)), dim=1)
            rendervar['scales'] = local_scales
        except: 
            rendervar['colors_precomp'] = local_colors
            rendervar['scales'] = local_scales
    return rendervar

# def transformed_GRNparams2depthplussilhouette(params, w2c, transformed_pts):
#     rendervar = {
#         'means3D': transformed_pts,
#         'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
#         'rotations': params['unnorm_rotations'],
#         'opacities': params['logit_opacities'],
#         # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     if params['log_scales'].shape[1] == 1:
#         rendervar['scales'] = torch.tile(params['log_scales'], (1, 3))
#     else:
#         rendervar['scales'] = params['log_scales']
#     return rendervar

def transformed_GRNparams2depthplussilhouette(params, w2c, transformed_pts,local_rots,local_scales,local_opacities):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(local_rots, dim =-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    if local_scales.shape[1] == 1:
        rendervar['scales'] = torch.tile(local_scales, (1, 3))
    else:
        rendervar['scales'] = local_scales
    return rendervar

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


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

def arap_loss_graph(params, base_nodes=None, arap_edges_k=6):
    """
    base_nodes: [Gm,3] rest positions (if None, use params['graph_nodes'].detach()).
    """
    if 'graph_nodes' not in params:
        return torch.tensor(0.0, device=params['means3D'].device)

    x0 = (base_nodes if base_nodes is not None else params['graph_nodes'].detach())
    # Current deformed node positions at *current* dt :
    # Reuse the most recent dt you used in deform_gaussians; or keep a scalar in params['timestep'] if you prefer.
    # For a simple, time-agnostic ARAP, we use the current learned translation 'node_t0' only:
    x = x0 + params['node_t0']  # small, stable variant

    with torch.no_grad():
        d2 = torch.cdist(x0, x0, p=2.0)
        knn_idx = torch.topk(d2, k=min(arap_edges_k+1, x0.shape[0]), dim=1, largest=False).indices[:, 1:]  # drop self

    # ARAP: sum || (x_i - x_j) - (x0_i - x0_j) ||
    xi = x.unsqueeze(1)                 # [Gm,1,3]
    xj = x[knn_idx]                     # [Gm,K,3]
    xi0 = x0.unsqueeze(1)               # [Gm,1,3]
    xj0 = x0[knn_idx]                   # [Gm,K,3]

    rig = ( (xi - xj) - (xi0 - xj0) )
    return (rig.pow(2).sum(-1).mean())
    
def arap_loss(params):
    """
    Simple ARAP: keep node-to-node edge vectors approximately rigid over time.
    """
    
    nodes = params['graph_nodes']
    with torch.no_grad():
        # 4-NN edges
        d = torch.cdist(nodes, nodes, p=2.0)
        knn = torch.topk(d, k=min(4, nodes.shape[0]-1), dim=1, largest=False).indices[:, 1:]  # [Gm,3]
    # current motion at dt read from a scalar params.get('last_dt', 0)
    dt = getattr(params, 'last_dt', torch.tensor(0.0, device=nodes.device))
    R_i, t_i, _ = _graph_node_motion(params, dt, use_fourier=('fourier_xyz_cos' in params))
    # warped neighbors (rigid part only)
    # For ARAP we compare rotated canonical edges
    v = (nodes[knn] - nodes[:, None, :])                          # [Gm,3,3]
    vR = (R_i[:, None, :, :] @ v.unsqueeze(-1)).squeeze(-1)       # [Gm,3,3]
    # encourage vR ≈ v
    return ((vR - v) ** 2).mean()


def deform_gaussians(params, time, return_all=False, deformation_type='svf', **kwargs):
    """
    SVF-only deformation.
    """
    n_squarings = int(params.get('svf_n_squarings', 8)) if isinstance(params, dict) else 8
    local_means = apply_svf(params, time, n_squarings=8)
    local_rots = params['unnorm_rotations']
    local_scales = params['log_scales']
    local_opacities = params['logit_opacities']
    local_colors = params['rgb_colors']
    if return_all:
        return local_means, local_rots, local_scales, local_opacities, local_colors
    return local_means, local_rots, local_scales, local_opacities, local_colors


# removed: initialize_cv_deformations



def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification,nr_basis = 10,use_distributed_biases = False, total_timescale = None,use_deform = True,deform_type = 'gaussian',num_frames = 1,
                            random_initialization = False,init_scale = 0.1):
    num_pts = new_pt_cld.shape[0]
    # Render all new gaussians with green color for debugging
    # new_pt_cld[:,3] = 0
    # new_pt_cld[:,5] = 0
    # new_pt_cld[:,4] = 1
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    if not random_initialization:
        params = {
            'means3D': means3D,
            'rgb_colors': new_pt_cld[:, 3:6],
            'unnorm_rotations': torch.tensor(unnorm_rots,dtype=torch.float).cuda(),
            'logit_opacities': logit_opacities,
            'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
        }
    else:
        params = {
            'means3D': means3D,
            'rgb_colors': new_pt_cld[:, 3:6],
            'unnorm_rotations': torch.zeros_like(torch.tensor(unnorm_rots),dtype=torch.float).cuda(),
            'logit_opacities': logit_opacities,
            'log_scales': torch.ones_like(means3D, dtype=torch.float, device=means3D.device) * init_scale,
        }
    # print(f'num pts {num_pts}')
    if use_deform and deform_type == 'gaussian':
        params = initialize_deformations(params, nr_basis=nr_basis,
                                     use_distributed_biases=use_distributed_biases,
                                     total_timescale=total_timescale)
        params = initialize_cv_deformations(params)
        params = initialize_graph_deformations(params=params)
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def grn_initialization(model,params,init_pt_cld,mean3_sq_dist,color,depth,mask = None,cam= None):

    normalize = torchvision.transforms.Normalize([0.46888983, 0.29536288, 0.28712815],[0.24689102 ,0.21034359, 0.21188641])
    inv_normalize = torchvision.transforms.Normalize([-0.46888983/0.24689102,-0.29536288/0.21034359,-0.28712815/0.21188641],[1/0.24689102,1/0.21034359,1/0.21188641]) #Take the inverse of the normalization

    color = normalize(color).detach()
    scale = torch.max(depth).detach()
    depth = (depth/scale).detach()

    input = torch.cat((color,depth),axis = 0).unsqueeze(0).detach()
    if mask == None:
        mask = depth > 0
        mask = mask.tile(1,3,1,1)
        mask = mask[0,0,:,:].reshape(-1)
        

    output = model(input).detach()

    # input is (1, 4, H, W); keep these to align GRN output
    _, _, H, W = input.shape
    output = model(input).detach()
    # bring GRN output to depth/color spatial size
    if output.shape[-2:] != depth.shape[-2:]:
        output = torch.nn.functional.interpolate(
            output, size=depth.shape[-2:], mode="bilinear", align_corners=False) 
    cols = output[0].permute(1, 2, 0).reshape(-1, 8)

    rots = cols[:,:4]

    # scales_norm = (cols[:,4:7]-cols[:,4:7].min()) / (cols[:,4:7].max()-cols[:,4:7].min())
    scales_norm = cols[:,4:7] 
    opacities = cols[:,7][:,None]

    # local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,0,False,5,'simple')
    # rendervar = transformed_GRNparams2rendervar(params,local_means,local_rots,local_scales,local_opacities,local_colors)   

    # im,radius,_ = Renderer(raster_settings=cam)(**rendervar)
    # plt.imshow(im.permute(1,2,0).cpu().detach())  
    # plt.title('Before grn_init')         
    # plt.show() 
    # If we use simple deformations, rotations and scales will have shape [C x Num_gaussians x num_frames],
    # We need to apply the GRN inialization to each timestep

    if len(params['unnorm_rotations'].shape) ==3:
        params['unnorm_rotations'] = (rots[mask])[...,None].tile(1,1,params['unnorm_rotations'].shape[2])
        params['log_scales'] = (scales_norm[mask]*(torch.sqrt(mean3_sq_dist)[:,None].tile(1,3)))[...,None].tile(1,1,params['log_scales'].shape[2])
        params['logit_opacities'] = (opacities[mask])[...,None].tile(1,1,params['logit_opacities'].shape[2])
    else:
        params['unnorm_rotations'] = rots[mask]
        params['log_scales'] = scales_norm[mask]*(torch.sqrt(mean3_sq_dist)[:,None].tile(1,3))
        params['logit_opacities'] = opacities[mask]
    

    # local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,0,False,5,'simple')
    # rendervar = transformed_GRNparams2rendervar(params,local_means,local_rots,local_scales,local_opacities,local_colors)    

    # im,radius,_ = Renderer(raster_settings=cam)(**rendervar)
    # plt.imshow(im.permute(1,2,0).cpu().detach())  
    # plt.title('After grn_init')         
    # plt.show() 


    return params

def get_mask(mask_input,color,reduction_type = 'random',reduction_fraction = 0.5):
    if reduction_type == 'random':
        mask = (torch.rand(mask_input.shape)>reduction_fraction).cuda()
        print('Reduction mask contains {} valid pixels, with reduction fraction of {}'.format(torch.sum(mask),reduction_fraction))
    else:
        mask = texture_mask_laplacian(color, num_samples=int(mask_input.sum()*(1-reduction_fraction))).cuda()
    return mask & mask_input

def texture_mask_laplacian(image_tensor: torch.Tensor, num_samples: int, ksize: int = 3):
    height = image_tensor.shape[1]
    width = image_tensor.shape[2]
    image_grayscale = torchvision.transforms.functional.rgb_to_grayscale(image_tensor)
    # Convert to NumPy uint8
    image_np = image_grayscale.squeeze().cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Apply Laplacian to get texture response
    laplacian = cv2.Laplacian(image_np, ddepth=cv2.CV_32F, ksize=ksize)
    texture_strength = np.abs(laplacian)

    # Normalize texture map to probability distribution
    texture_strength += 1e-6  # prevent division by zero
    prob_map = texture_strength / texture_strength.sum()

    # Flatten and sample
    flat_indices = np.arange(height * width)
    sampled_indices = np.random.choice(flat_indices, size=num_samples, replace=False, p=prob_map.flatten())

    # Create mask
    mask = np.zeros(height * width, dtype=np.uint8)
    mask[sampled_indices] = 1
    # mask = mask.reshape(height, width)

    return torch.from_numpy(mask).bool()

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True,
                      nr_basis = 10,use_distributed_biases = False,total_timescale = None,use_grn=False,grn_model=None,
                      use_deform = True,deformation_type = 'gaussian',num_frames = 1,
                      random_initialization=False,init_scale=0.1,cam = None, reduce_gaussians = False,reduction_type = 'random',reduction_fraction = 0.5):
    # Silhouette Rendering
    if use_deform == True:
        local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,time_idx,True,deformation_type =deformation_type)
    else:
        local_means = params['means3D']
        local_rots = params['unnorm_rotations']
        local_scales = params['log_scales']
        local_opacities = params['logit_opacities']
        local_colors = params['rgb_colors']
    
    transformed_pts = transform_to_frame(local_means,params, time_idx, gaussians_grad=False, camera_grad=False)
    if not use_grn:
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_pts,local_rots,local_scales,local_opacities)
    else:
        depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_pts,local_rots,local_scales,local_opacities)
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    # gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())*10+0.01
    


    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)
    if reduce_gaussians:
        mask = get_mask(non_presence_mask,color=curr_data['im'],reduction_type = reduction_type,reduction_fraction = reduction_fraction)

        non_presence_mask = non_presence_mask & mask
        
    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:


        # depth_diff = torch.abs(gt_depth - depth_sil[0, :, :])
        # fig,ax = plt.subplots(1,3)
        # im0 = ax[0].imshow(gt_depth.squeeze().cpu().detach())
        # ax[0].set_title('GT depth')
        # im1 = ax[1].imshow(depth_sil[0].squeeze().cpu().detach())
        # ax[1].set_title('Rendered depth')
        # im2 = ax[2].imshow(depth_diff.squeeze().cpu().detach())
        # ax[2].set_title('Depth diff')
        # plt.colorbar(im0,ax = ax[0])
        # plt.colorbar(im1,ax = ax[1])
        # plt.colorbar(im2,ax = ax[2])
        # plt.show()
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        print("Adding {} new gaussians".format(new_pt_cld.shape[0]))
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification,nr_basis = nr_basis,use_distributed_biases=use_distributed_biases,total_timescale = total_timescale,use_deform = use_deform,deform_type=deformation_type,
                                            num_frames = num_frames,random_initialization=random_initialization,init_scale=init_scale)
        

        # bloat_params = True
        # if bloat_params:
        #         new_params['bloated_params'] = torch.zeros(new_pt_cld.shape[0],1,device='cuda')
        #         # params_list = params
        if use_grn:
            new_params = grn_initialization(grn_model,new_params,new_pt_cld,mean3_sq_dist,curr_data['im'],curr_data['depth'],non_presence_mask,cam = cam)

        # # Adding new params happens to all timesteps due to construction of tensors, but they only need to be added to current and future timesteps,
        # # Therefore means,scales and rotations are set to 0 for previous timesteps
        # mask = torch.ones((1,1,new_params['means3D'].shape[-1]),device="cuda")
        # mask[0,0,time_idx-1:] = 0
        params_iter = {}
        for k, v in new_params.items():
            params_iter[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        params_iter['cam_unnorm_rots'] = params['cam_unnorm_rots']
        params_iter['cam_trans'] = params['cam_trans']
        params_iter['svf'] = params['svf']
        params_iter['svf_aabb_center'] = params['svf_aabb_center']
        params_iter['svf_aabb_half'] = params['svf_aabb_half']
        params_iter['svf_L0'] = params['svf_L0']
        params_iter['svf_L1'] = params['svf_L1']
        params_iter['svf_n_squarings'] = params['svf_n_squarings']
        num_pts = params_iter['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    else:
        params_iter = params
    return params_iter, variables

def align_shift_and_scale(gt_disp, pred_disp, mask):

    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if valid.sum() == 0:
        # Return input as-is or raise a warning
        print("⚠️ Warning: Empty valid mask in align_shift_and_scale")
        return gt_disp, pred_disp, torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])
    
    # Select valid pixels
    gt_selected = gt_disp[mask].view(valid.sum(), -1)
    pred_selected = pred_disp[mask].view(valid.sum(), -1)

    # Compute statistics
    t_gt = torch.median(gt_selected, dim=1).values
    s_gt = torch.mean(torch.abs(gt_selected - t_gt[:, None]), dim=1)

    t_pred = torch.median(pred_selected, dim=1).values
    s_pred = torch.mean(torch.abs(pred_selected - t_pred[:, None]), dim=1)

    # Normalize
    pred_disp_aligned = ((pred_disp.view(pred_disp.shape[0], -1) - t_pred[:, None]) / s_pred[:, None])
    pred_disp_aligned = pred_disp_aligned.view_as(pred_disp)

    gt_disp_aligned = ((gt_disp.view(gt_disp.shape[0], -1) - t_gt[:, None]) / s_gt[:, None])
    gt_disp_aligned = gt_disp_aligned.view_as(gt_disp)

    return gt_disp_aligned, pred_disp_aligned, t_gt, s_gt, t_pred, s_pred

"""
def align_shift_and_scale(gt_disp, pred_disp,mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    
    t_gt = torch.median((gt_disp[mask]*mask[mask]).view(valid.sum(),-1),dim = 1).values
    # print(t_gt)
    # print(gt_disp[valid].view(valid.sum(),-1).shape,t_gt.shape)

    s_gt = torch.mean(torch.abs(gt_disp[mask].view(valid.sum(),-1)- t_gt[:,None]),1)
    t_pred = torch.median((pred_disp[mask]*mask[mask]).view(valid.sum(),-1),dim = 1).values
    s_pred = torch.mean(torch.abs(pred_disp[mask].view(valid.sum(),-1)- t_pred[:,None]),1)
    # print(pred_disp.view(gt_disp.shape[0],-1).shape,t_pred.shape,s_pred.shape)
    pred_disp_aligned = (pred_disp.view(pred_disp.shape[0],-1)- t_pred[:,None])/s_pred[:,None]
    pred_disp_aligned = pred_disp_aligned.view(pred_disp.shape[0],pred_disp.shape[1],pred_disp.shape[2])


    gt_disp_aligned = (gt_disp.view(gt_disp.shape[0],-1)- t_gt[:,None])/s_gt[:,None]
    gt_disp_aligned = gt_disp_aligned.view(gt_disp.shape[0],gt_disp.shape[1],gt_disp.shape[2])
    return  gt_disp_aligned, pred_disp_aligned,t_gt, s_gt, t_pred, s_pred

    """

# ========================= SVF (Stationary Velocity Field) Deformer =========================
class SVFDeformer(torch.nn.Module):
    """
    Multi-level stationary velocity field over an AABB. Each level is a 3D grid (C=3).
    Coordinates are mapped from world space to [-1,1]^3 using params['svf_aabb'] = (center[3], half[3]).
    """
    def __init__(self, levels=(32, 64)):
        super().__init__()
        self.levels = list(levels)
        # Create learnable velocity fields per level: [1,3,D,H,W]
        fields = []
        for i, d in enumerate(self.levels):
            p = torch.nn.Parameter(torch.zeros(1, 3, d, d, d, dtype=torch.float32))
            self.register_parameter(f"svf_L{i}", p)
            fields.append(p)
        self._fields = fields  # keep refs for iteration

    def iter_fields(self):
        for i, _ in enumerate(self.levels):
            yield getattr(self, f"svf_L{i}")

    def forward(self, x_world, center, half):
        """
        x_world: [G,3] world coordinates
        center: [3], half: [3] defining the AABB
        returns v(x): [G,3]
        """
        # normalize to [-1,1]
        x = (x_world - center[None, :]) / (half[None, :] + 1e-8)
        x = torch.clamp(x, -1.5, 1.5)  # allow mild extrapolation
        # Sample each level with grid_sample; build a fake "volume" of points
        pts = x.view(1, 1, x.shape[0], 1, 3)  # [N=1, D'=1, H'=G, W'=1, 3]
        v = 0.0
        for F_l in self.iter_fields():
            v_l = torch.nn.functional.grid_sample(
                F_l, pts, mode="bilinear", padding_mode="border", align_corners=True
            )  # [1,3,1,G,1]
            v_l = v_l.view(3, x.shape[0]).T  # [G,3]
            v = v + v_l
        return v

def initialize_svf(params, levels=(32,64)):
    """
    Create an SVFDeformer and attach params:
      - params['svf'] = SVFDeformer(levels)
      - params['svf_aabb_center'], params['svf_aabb_half']
      - Also expose each level tensor as a top-level param 'svf_L{i}' so your existing optimizer picks it up.
    """
    
    means = params['means3D'].detach()
    center = means.mean(0)
    span = means.max(0).values - means.min(0).values
    half = span.clamp(min=1e-2) * 0.75  # slightly inside the current spread

    svf = SVFDeformer(levels=levels).to(means.device)
    params['svf'] = svf
    params['svf_aabb_center'] = torch.nn.Parameter(center.clone(), requires_grad=False)
    params['svf_aabb_half'] = torch.nn.Parameter(half.clone(), requires_grad=False)
    # expose fields at top-level for optimizer lr mapping
    for i, F_l in enumerate(svf.iter_fields()):
        params[f'svf_L{i}'] = F_l
    # optional: number of squarings
    if 'svf_n_squarings' not in params:
        params['svf_n_squarings'] = 8
    return params

@torch.no_grad()
def update_svf_aabb(params, momentum=0.05):
    """Slowly update the SVF AABB to follow the cloud without jitter."""
    means = params['means3D'].detach()
    center = means.mean(0)
    span = means.max(0).values - means.min(0).values
    half = span.clamp(min=1e-2) * 0.75
    # EMA
    params['svf_aabb_center'].data = (1 - momentum) * params['svf_aabb_center'].data + momentum * center
    params['svf_aabb_half'].data   = (1 - momentum) * params['svf_aabb_half'].data   + momentum * half

def apply_svf(params, time_idx=None, n_squarings=8):
    """
    Stationary velocity field integration (scaling-and-squaring style Euler).
    Deforms Gaussian means in *world* coordinates. Other attributes unchanged.

    Tunables (optional, set in params):
      - params['svf_scale']           : float factor to amplify the velocity (default 3.0)
      - params['svf_n_squarings']     : int number of squarings / Euler refinements (default 8)
      - params['svf_step_clip_ratio'] : max per-step |u| as a fraction of AABB half extent (default 0.05)
    """
    assert 'svf' in params, "SVF not initialized. Call initialize_svf."

    # --- read hyperparams from params when available
    # n_squarings may be stored as tensor; make it a python int
    if 'svf_n_squarings' in params:
        try:
            n_squarings = int(params['svf_n_squarings'])
        except Exception:
            n_squarings = int(getattr(params['svf_n_squarings'], 'item', lambda: 8)())
    svf_scale = float(3.0)
    step_clip_ratio = float(0.05)  # 5% of box size per substep

    x0     = params['means3D']                      # (N,3) world coords
    center = params['svf_aabb_center']              # (3,)
    half   = params['svf_aabb_half']                # (3,)

    # size-based clip to avoid folding; scalar in world units
    # (use largest half-extent so it's safe in all axes)
    step_clip = step_clip_ratio * torch.as_tensor(half, device=x0.device, dtype=x0.dtype).abs().max()

    # scale per substep (classic scaling-and-squaring)
    substep_scale = svf_scale / (2.0 ** n_squarings)

    # first substep
    u = params['svf'](x0, center, half) * substep_scale
    if step_clip > 0:
        u = u.clamp(min=-step_clip, max=step_clip)
    y = x0 + u

    # refine with additional squarings (Euler composition)
    for _ in range(n_squarings):
        u = params['svf'](y, center, half) * substep_scale
        if step_clip > 0:
            u = u.clamp(min=-step_clip, max=step_clip)
        y = y + u

    return y

def svf_bending_loss(params):
    """
    Bending energy ||∇v||^2 over all levels.
    """
    if 'svf' not in params:
        return torch.tensor(0.0, device=params['means3D'].device if 'means3D' in params else 'cpu')

    loss = 0.0
    for p in params['svf'].iter_fields():
        # finite diffs
        dx = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :]).pow(2).mean()
        dy = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :]).pow(2).mean()
        dz = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1]).pow(2).mean()
        loss = loss + (dx + dy + dz)
    return loss
# ============================================================================================
