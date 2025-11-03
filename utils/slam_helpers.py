import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import energy_mask
import torchvision
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from utils.deform_mlp import TimeDeformMLP


def _identity_init(params: Dict[str, torch.Tensor], **_: Any) -> Dict[str, torch.Tensor]:
    return params


def _noop_finalize(container: Any, **_: Any) -> Any:
    return container


def _noop_snapshot(_: Any) -> Optional[Dict[str, torch.Tensor]]:
    return None


def _noop_restore(container: Any, snapshot: Optional[Dict[str, torch.Tensor]]) -> Any:
    return container


def _default_container_builder(params: Dict[str, torch.Tensor], **_: Any) -> Dict[str, torch.Tensor]:
    return params


@dataclass
class DeformationModel:
    name: str
    apply: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    init_scene: Callable[..., Dict[str, torch.Tensor]] = field(default=_identity_init)
    init_new: Callable[..., Dict[str, torch.Tensor]] = field(default=_identity_init)
    snapshot: Callable[[Any], Optional[Dict[str, torch.Tensor]]] = field(default=_noop_snapshot)
    restore: Callable[[Any, Optional[Dict[str, torch.Tensor]]], Any] = field(default=_noop_restore)
    finalize: Callable[..., Any] = field(default=_noop_finalize)
    build_container: Callable[..., Any] = field(default=_default_container_builder)


_DEFORMATION_REGISTRY: Dict[str, DeformationModel] = {}


def register_deformation_model(model: DeformationModel, *, aliases: Iterable[str] = ()) -> None:
    """Registers a deformation model and optional aliases."""
    _DEFORMATION_REGISTRY[model.name] = model
    for alias in aliases:
        _DEFORMATION_REGISTRY[alias] = model


def get_deformation_model(name: str) -> DeformationModel:
    if name not in _DEFORMATION_REGISTRY:
        available = ", ".join(sorted(_DEFORMATION_REGISTRY.keys()))
        raise KeyError(f"Unknown deformation model '{name}'. Available: {available}")
    return _DEFORMATION_REGISTRY[name]


def list_deformation_models() -> Tuple[str, ...]:
    return tuple(sorted(_DEFORMATION_REGISTRY.keys()))


def _snapshot_by_keys(params: Dict[str, torch.Tensor], keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    snapshot: Dict[str, torch.Tensor] = {}
    for key in keys:
        if key in params and isinstance(params[key], torch.Tensor):
            snapshot[key] = params[key].detach().clone()
    return snapshot


def _restore_by_keys(params: Dict[str, torch.Tensor], snapshot: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if snapshot is None:
        return params
    for key, value in snapshot.items():
        if key in params and isinstance(params[key], torch.Tensor):
            params[key].data.copy_(value)
    return params


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _scales_to_flat_matrix(
    scales: torch.Tensor,
    *,
    exp_input: bool = True,
    clamp_min: float = 1e-3,
    clamp_max: float = 0.25,
) -> torch.Tensor:
    """
    Convert per-gaussian scale parameters into the format expected by the renderer.
    The renderer in this environment consumes diagonal covariances (3 values).
    Fall back gracefully if a 3x3 representation sneaks in.
    """
    if exp_input:
        scales = torch.exp(scales)

    if scales.ndim == 1:
        scales = scales.view(1, -1)
    elif scales.ndim > 2:
        scales = scales.view(scales.shape[0], -1)

    last_dim = scales.shape[-1]
    if last_dim == 1:
        scales = scales.expand(-1, 3)
    elif last_dim == 3:
        pass
    elif last_dim == 9:
        # Renderer only accepts diagonal entries; pick them out.
        scales = scales.view(-1, 3, 3).diagonal(dim1=1, dim2=2)
    else:
        raise ValueError(f"Unexpected scale shape {tuple(scales.shape)}")

    return torch.clamp(scales, min=clamp_min, max=clamp_max)


def _apply_endo4dgs_model(
    params: Dict[str, torch.Tensor],
    *,
    time: float,
    deform_grad: bool,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if deform_grad:
        base_xyz = params['means3D']
        base_rots = params['unnorm_rotations']
        base_scl = params['log_scales']
        base_op = params['logit_opacities']
        base_col = params['rgb_colors']
    else:
        base_xyz = params['means3D'].detach()
        base_rots = params['unnorm_rotations'].detach()
        base_scl = params['log_scales'].detach()
        base_op = params['logit_opacities'].detach()
        base_col = params['rgb_colors'].detach()

    deform_net = params.get('endo4dgs_net', None)
    if deform_net is None:
        rots = F.normalize(base_rots, dim=-1)
        return base_xyz, rots, base_scl, base_op, base_col

    if isinstance(deform_net, torch.nn.Parameter):
        # it got wrapped somewhere; rebuild it
        params = rehydrate_endo4dgs(
            params,
            config={},  # or params.get('endo4dgs_net_meta', {})
            device=base_xyz.device
        )
        deform_net = params['endo4dgs_net']

    device = base_xyz.device
    dtype = base_xyz.dtype
    t_scalar = torch.as_tensor(time, device=device, dtype=dtype)
    if deform_grad:
        d_xyz, d_log_s, d_rot_aa, d_op = deform_net(base_xyz, t_scalar)
    else:
        with torch.no_grad():
            d_xyz, d_log_s, d_rot_aa, d_op = deform_net(base_xyz, t_scalar)

    # Safety checks to prevent NaN
    xyz = base_xyz + d_xyz
    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=1e6, neginf=-1e6)
    
    scales = base_scl + d_log_s
    scales = torch.clamp(scales, min=-5.0, max=5.0)  # Prevent extreme scales

    delta_q = _aa_to_quat(d_rot_aa)
    base_q = F.normalize(base_rots, dim=-1)
    rots = F.normalize(_qmul(base_q, delta_q), dim=-1)
    rots = torch.nan_to_num(rots, nan=0.0)
    # Re-normalize to ensure valid quaternion
    rots = F.normalize(rots, dim=-1)

    opacities = base_op + d_op
    opacities = torch.clamp(opacities, min=-8.0, max=8.0)  # Keep in reasonable logit range
    
    colors = base_col
    return xyz, rots, scales, opacities, colors


def _snapshot_endo4dgs(params: Dict[str, Any], **_: Any) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    net = params.get('endo4dgs_net', None)
    if net is None:
        return None
    return {'endo4dgs_net': _clone_state_dict(net.state_dict())}


def _restore_endo4dgs(
    params: Dict[str, Any],
    snapshot: Optional[Dict[str, Dict[str, torch.Tensor]]],
    **_: Any,
) -> Dict[str, Any]:
    if snapshot is None:
        return params
    state = snapshot.get('endo4dgs_net')
    net = params.get('endo4dgs_net', None)
    if state is not None and net is not None:
        net.load_state_dict(state)
    return params


def _finalize_endo4dgs(container: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    net = container.pop('endo4dgs_net', None)
    if net is not None:
        state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        container['endo4dgs_net_state'] = np.array([state], dtype=object)
        container['endo4dgs_net_meta'] = np.array(
            [{
                'in_dim': getattr(net, 'input_dim', getattr(net, 'in_features', None)),
                'hidden': getattr(net, 'hidden_dim', getattr(net, 'out_features', None)),
            }],
            dtype=object,
        )
    return container


def rehydrate_endo4dgs(
    params: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Any] = None,
    meta: Optional[Any] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    if config is None:
        config = {}
    hidden = int(config.get('mlp_hidden', 64))
    in_dim = int(config.get('mlp_in_dim', 4))

    if meta is None and 'endo4dgs_net_meta' in params:
        meta = params.pop('endo4dgs_net_meta')
    if isinstance(meta, np.ndarray) and meta.size > 0:
        meta = meta.item()
    if isinstance(meta, dict):
        hidden = int(meta.get('hidden', hidden) or hidden)
        in_dim = int(meta.get('in_dim', in_dim) or in_dim)

    if state is None and 'endo4dgs_net_state' in params:
        state = params.pop('endo4dgs_net_state')
    if isinstance(state, np.ndarray) and state.size > 0:
        state = state.item()

    net = TimeDeformMLP(in_dim=in_dim, hidden=hidden)
    if isinstance(state, dict):
        state_tensors = {k: torch.from_numpy(v).float() for k, v in state.items()}
        net.load_state_dict(state_tensors, strict=False)
    params['endo4dgs_net'] = net.to(device)
    return params


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
    
    scales_flat = _scales_to_flat_matrix(local_scales, exp_input=True)
    if local_scales.shape[0] == 1:
        rendervar['colors_precomp'] = local_colors
    else:
        try:
            rendervar['shs'] = torch.cat(
                (
                    local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
                    params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2),
                ),
                dim=1,
            )
        except Exception:
            rendervar['colors_precomp'] = local_colors
    rendervar['scales'] = scales_flat
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


def get_depth_and_silhouette(pts_3D, w2c=None):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    if w2c is not None:
        pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
        pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    else:
        pts_in_cam = pts_3D
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    device = pts_3D.device
    dtype = pts_3D.dtype
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3), device=device, dtype=dtype)
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = torch.as_tensor(1.0, device=device, dtype=dtype)
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


def transformed_params2depthplussilhouette(params, w2c, transformed_pts,local_rots,local_scales,local_opacities, *, camera_space: bool = False):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, None if camera_space else w2c),
        'rotations': F.normalize(local_rots, dim=-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    rendervar['scales'] = _scales_to_flat_matrix(local_scales, exp_input=True)
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
            if gt.numel() == 0:
                # No Gaussians to gate - return empty render_vars
                return render_vars
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
    scales_flat = _scales_to_flat_matrix(local_scales, exp_input=False)
    if local_scales.shape[0] == 1:
        rendervar['colors_precomp'] = local_colors
    else:
        try:
            rendervar['shs'] = torch.cat((local_colors.reshape(local_colors.shape[0], 3, -1).transpose(1, 2), params['feature_rest'].reshape(local_colors.shape[0], 3, -1).transpose(1, 2)), dim=1)
        except:
            rendervar['colors_precomp'] = local_colors
    rendervar['scales'] = scales_flat
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

def transformed_GRNparams2depthplussilhouette(params, w2c, transformed_pts,local_rots,local_scales,local_opacities, *, camera_space: bool = False):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, None if camera_space else w2c),
        'rotations': F.normalize(local_rots, dim =-1),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    rendervar['scales'] = _scales_to_flat_matrix(local_scales, exp_input=False)
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

def _apply_gaussian_model(
    params: Dict[str, torch.Tensor],
    *,
    time: float,
    deform_grad: bool,
    N: Optional[int] = 5,
    temperature: float = 0.5,
    detach_canonical: bool = False,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if deform_grad or not detach_canonical:
        base_xyz = params['means3D']
        base_rots = params['unnorm_rotations']
        base_scl = params['log_scales']
        base_op = params['logit_opacities']
        base_col = params['rgb_colors']
    else:
        base_xyz = params['means3D'].detach()
        base_rots = params['unnorm_rotations'].detach()
        base_scl = params['log_scales'].detach()
        base_op = params['logit_opacities'].detach()
        base_col = params['rgb_colors'].detach()

    if deform_grad:
        W = params['deform_weights']
        S_raw = params['deform_stds']
        C = params['deform_biases']
    else:
        W = params['deform_weights'].detach()
        S_raw = params['deform_stds'].detach()
        C = params['deform_biases'].detach()

    t = torch.as_tensor(time, device=C.device, dtype=C.dtype)
    while t.dim() < C.dim():
        t = t.unsqueeze(0)

    S = F.softplus(S_raw) + 1e-3
    score = -((t - C) ** 2) / (2.0 * (S ** 2) + 1e-12)
    attn = F.softmax(score / max(temperature, 1e-3), dim=1)

    if N is not None and isinstance(N, int) and 0 < N < attn.shape[1]:
        topv, topi = torch.topk(attn, k=N, dim=1)
        m = torch.zeros_like(attn)
        m.scatter_(1, topi, 1.0)
        attn = attn * m
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-12)

    deform = torch.sum(attn * W, dim=1)
    d_xyz = deform[:, 0:3]
    d_quat = deform[:, 3:7]
    d_scl = deform[:, 7:10]

    xyz = base_xyz + d_xyz

    base_q = F.normalize(base_rots, dim=-1)
    delta_q = F.normalize(d_quat, dim=-1)
    rots = F.normalize(_qmul(base_q, delta_q), dim=-1)

    scales = base_scl + d_scl
    return xyz, rots, scales, base_op, base_col


def _apply_simple_model(
    params: Any,
    *,
    time: int,
    deform_grad: bool,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del deform_grad  # simple model ignores gradients here
    if isinstance(params, (list, tuple)):
        params_t = params[int(time)]
    else:
        params_t = params
    xyz = params_t['means3D']
    rots = params_t['unnorm_rotations']
    scales = params_t['log_scales']
    opacities = params_t['logit_opacities']
    colors = params_t['rgb_colors']
    return xyz, rots, scales, opacities, colors


def _apply_cv_model(
    params: Dict[str, torch.Tensor],
    *,
    time: float,
    deform_grad: bool,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del deform_grad
    means = params['means3D']
    t = torch.as_tensor(time, device=means.device, dtype=means.dtype).view(-1, 1)

    v_xyz = params.get('cv_vel_xyz', torch.zeros_like(means))
    v_lsg = params.get('cv_vel_log_scales', torch.zeros_like(params['log_scales']))
    w_aa = params.get('cv_angvel_aa', torch.zeros_like(means))

    xyz = params['means3D'] + v_xyz * t
    base_q = F.normalize(params['unnorm_rotations'], dim=-1)
    dq = _aa_to_quat(w_aa * t)
    rots = F.normalize(_qmul(base_q, dq), dim=-1)
    scales = params['log_scales'] + v_lsg * t

    opacities = params['logit_opacities']
    colors = params['rgb_colors']
    return xyz, rots, scales, opacities, colors


def deform_gaussians(
    params,
    time,
    deform_grad,
    N=5,
    deformation_type='gaussian',
    temperature=0.5,
    **kwargs: Any,
):
    """
    Apply the requested deformation model through the registry.
    """
    model = get_deformation_model(deformation_type)
    return model.apply(
        params,
        time=time,
        deform_grad=deform_grad,
        N=N,
        temperature=temperature,
        deformation_type=deformation_type,
        **kwargs,
    )


def initialize_cv_deformations(params):
    """Adds per-Gaussian constant-velocity params (very memory efficient)."""
    G      = params['means3D'].shape[0]
    device = params['means3D'].device
    zeros3 = torch.zeros(G, 3, device=device)

    params['cv_vel_xyz']        = torch.nn.Parameter(zeros3.clone(), requires_grad=True)
    params['cv_vel_log_scales'] = torch.nn.Parameter(zeros3.clone(), requires_grad=True)
    params['cv_angvel_aa']      = torch.nn.Parameter(zeros3.clone(), requires_grad=True)
    return params


def initialize_deformations(params, nr_basis, use_distributed_biases, total_timescale=None):
    """Initialize Gaussian-basis deformation parameters (lean on memory)."""
    N = params['means3D'].shape[0]
    device = 'cuda'

    # weights/stds need grads; biases often don't
    weights = torch.randn([N, nr_basis, 10], device=device, dtype=torch.float16) * 0.0
    stds    = torch.ones([N, nr_basis, 10],  device=device, dtype=torch.float16) / 0.1

    weights = torch.nn.Parameter(weights, requires_grad=True)
    stds    = torch.nn.Parameter(stds,    requires_grad=True)

    if not use_distributed_biases:
        biases = torch.randn([N, nr_basis, 10], device=device, dtype=torch.float16) * 0.0
        biases = torch.nn.Parameter(biases, requires_grad=True)
    else:
        max_time = float(total_timescale if total_timescale is not None else nr_basis)
        if max_time <= 1.0:
            centers = torch.zeros(nr_basis, device=device, dtype=torch.float16)
        else:
            centers = torch.linspace(
                0.0,
                max_time - 1.0,
                steps=nr_basis,
                device=device,
                dtype=torch.float16,
            )
        biases = centers.view(1, nr_basis, 1).repeat(N, 1, 10)
        biases = torch.nn.Parameter(biases, requires_grad=False)

    params['deform_weights'] = weights
    params['deform_stds']    = stds
    params['deform_biases']  = biases
    return params


def _init_basis_params(
    params: Dict[str, torch.Tensor],
    *,
    deform_cfg: Optional[Dict[str, Any]] = None,
    total_timescale: Optional[int] = None,
    **_: Any,
) -> Dict[str, torch.Tensor]:
    deform_cfg = deform_cfg or {}
    nr_basis = int(deform_cfg.get('nr_basis', 10))
    use_distributed = bool(deform_cfg.get('use_distributed_biases', False))
    if total_timescale is None:
        total_timescale = deform_cfg.get('total_timescale', None)
    return initialize_deformations(
        params,
        nr_basis=nr_basis,
        use_distributed_biases=use_distributed,
        total_timescale=total_timescale,
    )


def _init_cv_params(
    params: Dict[str, torch.Tensor],
    **_: Any,
) -> Dict[str, torch.Tensor]:
    return initialize_cv_deformations(params)


def _snapshot_simple(container: Any) -> Optional[Any]:
    if isinstance(container, (list, tuple)):
        snapshots = []
        for frame_params in container:
            if isinstance(frame_params, dict):
                snapshots.append(
                    _snapshot_by_keys(
                        frame_params,
                        ('means3D', 'unnorm_rotations', 'log_scales'),
                    )
                )
        return snapshots
    if isinstance(container, dict):
        return _snapshot_by_keys(
            container,
            ('means3D', 'unnorm_rotations', 'log_scales'),
        )
    return None


def _restore_simple(container: Any, snapshot: Optional[Any]) -> Any:
    if snapshot is None:
        return container
    if isinstance(container, (list, tuple)) and isinstance(snapshot, list):
        for params_dict, snap in zip(container, snapshot):
            if isinstance(params_dict, dict) and isinstance(snap, dict):
                _restore_by_keys(params_dict, snap)
        return container
    if isinstance(container, dict) and isinstance(snapshot, dict):
        _restore_by_keys(container, snapshot)
    return container


def _build_simple_container(
    base_params: Dict[str, torch.Tensor],
    *,
    num_frames: int,
    per_frame_keys: Tuple[str, ...] = (
        'means3D',
        'unnorm_rotations',
        'log_scales',
        'logit_opacities',
        'rgb_colors',
    ),
    **_: Any,
) -> Any:
    frame_params_list = []
    for _ in range(num_frames):
        frame_params: Dict[str, torch.Tensor] = {}
        for key, value in base_params.items():
            if key in per_frame_keys and isinstance(value, torch.Tensor):
                cloned = torch.nn.Parameter(value.detach().clone(), requires_grad=value.requires_grad)
                frame_params[key] = cloned
            else:
                frame_params[key] = value
        frame_params_list.append(frame_params)
    return frame_params_list


def _finalize_simple(container: Any, **_: Any) -> Any:
    if not isinstance(container, (list, tuple)) or len(container) == 0:
        return container
    params_save: Dict[str, Any] = {}
    keys_to_stack = ('means3D', 'unnorm_rotations', 'log_scales', 'logit_opacities', 'rgb_colors')
    for key in keys_to_stack:
        params_save[key] = [frame[key] for frame in container]
    ref = container[-1]
    params_save['cam_unnorm_rots'] = ref.get('cam_unnorm_rots', None)
    params_save['cam_trans'] = ref.get('cam_trans', None)
    return params_save


register_deformation_model(
    DeformationModel(
        name='gaussian',
        apply=_apply_gaussian_model,
        init_scene=_init_basis_params,
        init_new=_init_basis_params,
        snapshot=lambda params, **_: _snapshot_by_keys(
            params,
            ('deform_weights', 'deform_stds', 'deform_biases'),
        ),
        restore=lambda params, snap, **_: _restore_by_keys(params, snap),
    )
)

register_deformation_model(
    DeformationModel(
        name='deformgs',
        apply=lambda params, **kwargs: _apply_gaussian_model(
            params,
            detach_canonical=True,
            **kwargs,
        ),
        init_scene=_init_basis_params,
        init_new=_init_basis_params,
        snapshot=lambda params, **_: _snapshot_by_keys(
            params,
            ('deform_weights', 'deform_stds', 'deform_biases'),
        ),
        restore=lambda params, snap, **_: _restore_by_keys(params, snap),
    )
)

register_deformation_model(
    DeformationModel(
        name='cv',
        apply=_apply_cv_model,
        init_scene=_init_cv_params,
        init_new=_init_cv_params,
        snapshot=lambda params, **_: _snapshot_by_keys(
            params,
            ('cv_vel_xyz', 'cv_vel_log_scales', 'cv_angvel_aa'),
        ),
        restore=lambda params, snap, **_: _restore_by_keys(params, snap),
    )
)

register_deformation_model(
    DeformationModel(
        name='endo4dgs',
        apply=_apply_endo4dgs_model,
        snapshot=_snapshot_endo4dgs,
        restore=_restore_endo4dgs,
        finalize=_finalize_endo4dgs,
    )
)

register_deformation_model(
    DeformationModel(
        name='simple',
        apply=_apply_simple_model,
        snapshot=_snapshot_simple,
        restore=_restore_simple,
        build_container=_build_simple_container,
        finalize=_finalize_simple,
    )
)


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
            'log_scales': torch.ones_like(torch.tensor(means3D),dtype=torch.float).cuda()*init_scale,
        }
    # print(f'num pts {num_pts}')
    if use_deform:
        deform_cfg = {
            'nr_basis': nr_basis,
            'use_distributed_biases': use_distributed_biases,
            'total_timescale': total_timescale,
            'num_frames': num_frames,
        }
        model = get_deformation_model(deform_type)
        params = model.init_new(
            params,
            deform_cfg=deform_cfg,
            total_timescale=total_timescale,
            num_frames=num_frames,
        )
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
    MAX_NEW_PER_FRAME = 2000  # hard cap to stop densification bursts from OOM'ing
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
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            params,
            curr_data['w2c'],
            transformed_pts,
            local_rots,
            local_scales,
            local_opacities,
            camera_space=True,
        )
    else:
        depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(
            params,
            curr_data['w2c'],
            transformed_pts,
            local_rots,
            local_scales,
            local_opacities,
            camera_space=True,
        )
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
    total_pixels = non_presence_mask.numel()
    # fraction of image pixels flagged as lacking coverage this frame
    missing_ratio = float(non_presence_mask.sum().item()) / float(total_pixels) if total_pixels > 0 else 0.0
        
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
        point_indices = torch.nonzero(non_presence_mask, as_tuple=False).squeeze(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        candidate_new = new_pt_cld.shape[0]
        if candidate_new > MAX_NEW_PER_FRAME:
            keep_idx = torch.randperm(candidate_new, device=new_pt_cld.device)[:MAX_NEW_PER_FRAME]
            new_pt_cld = new_pt_cld[keep_idx]
            mean3_sq_dist = mean3_sq_dist[keep_idx]
            point_indices = point_indices[keep_idx]
            print(f"[densify] capping per-frame additions from {candidate_new} to {MAX_NEW_PER_FRAME}")
        original_new = new_pt_cld.shape[0]
        print("Adding {} new gaussians".format(original_new))

        curr_N = params['means3D'].shape[0]
        base_cap = 4000
        budget = max(0, 50000 - curr_N)
        max_new = max(0, min(base_cap, budget))
        if max_new == 0:
            print("[densify] budget exhausted; skipping new gaussians")
            return params, variables, missing_ratio
        n_new = new_pt_cld.shape[0]
        if n_new > max_new:
            sort_idx = torch.argsort(mean3_sq_dist)[:max_new]
            new_pt_cld = new_pt_cld[sort_idx]
            mean3_sq_dist = mean3_sq_dist[sort_idx]
            n_new = max_new
            point_indices = point_indices[sort_idx]
            print(f'[densify] clipped from {original_new} to {max_new}')

        effective_nr_basis = nr_basis
        effective_use_distributed = use_distributed_biases
        if n_new > 1500 and nr_basis > 4:
            effective_nr_basis = 4
            effective_use_distributed = True

        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification,
                                           nr_basis=effective_nr_basis,
                                           use_distributed_biases=effective_use_distributed,
                                           total_timescale=total_timescale,use_deform = use_deform,
                                           deform_type=deformation_type,
                                           num_frames = num_frames,random_initialization=random_initialization,
                                           init_scale=init_scale)
        

        # bloat_params = True
        # if bloat_params:
        #         new_params['bloated_params'] = torch.zeros(new_pt_cld.shape[0],1,device='cuda')
        #         # params_list = params
        if use_grn:
            selected_mask = torch.zeros_like(non_presence_mask, dtype=torch.bool)
            if point_indices.numel() > 0:
                selected_mask[point_indices] = True
            new_params = grn_initialization(grn_model,new_params,new_pt_cld,mean3_sq_dist,curr_data['im'],curr_data['depth'],selected_mask,cam = cam)

        if 'logit_opacities' in new_params:
            with torch.no_grad():
                if n_new > 2000:
                    new_params['logit_opacities'].data.clamp_(min=-4.0, max=-2.0)
                else:
                    new_params['logit_opacities'].data.clamp_(min=-3.5, max=-1.5)

        # # Adding new params happens to all timesteps due to construction of tensors, but they only need to be added to current and future timesteps,
        # # Therefore means,scales and rotations are set to 0 for previous timesteps
        # mask = torch.ones((1,1,new_params['means3D'].shape[-1]),device="cuda")
        # mask[0,0,time_idx-1:] = 0
        params_iter = {}
        for k, v in new_params.items():
            prev_vals = params[k].detach()
            new_vals = v.detach()
            combined = torch.cat((prev_vals, new_vals), dim=0)
            params_iter[k] = torch.nn.Parameter(combined)
        params_iter['cam_unnorm_rots'] = params['cam_unnorm_rots']
        params_iter['cam_trans'] = params['cam_trans']
        if 'endo4dgs_net' in params:
            params_iter['endo4dgs_net'] = params['endo4dgs_net']
        num_pts = params_iter['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
        if 'last_seen' in variables:
            new_last_seen = time_idx * torch.ones(new_pt_cld.shape[0], device="cuda").float()
            variables['last_seen'] = torch.cat((variables['last_seen'], new_last_seen), dim=0)
        if 'visibility_hits' in variables:
            new_hits = torch.zeros(new_pt_cld.shape[0], device="cuda").float()
            variables['visibility_hits'] = torch.cat((variables['visibility_hits'], new_hits), dim=0)
    else:
        params_iter = params
        missing_ratio = 0.0
    return params_iter, variables, missing_ratio

def align_shift_and_scale(gt_disp, pred_disp, mask):

    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    device = gt_disp.device
    dtype = gt_disp.dtype

    if valid.sum() == 0:
        # Return input as-is or raise a warning
        print("⚠️ Warning: Empty valid mask in align_shift_and_scale")
        zero = torch.zeros(1, device=device, dtype=dtype)
        one = torch.ones(1, device=device, dtype=dtype)
        return gt_disp, pred_disp, zero, one, zero.clone(), one.clone()
    
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
