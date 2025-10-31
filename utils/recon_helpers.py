import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import numpy as np


def setup_camera(w, h, k, w2c, near=0.01, far=100, bg=[0,0,0], use_simplification=True):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0 if use_simplification else 3,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def calculate_entropy(prob):
    return -np.sum(prob * np.log2(prob + np.finfo(float).eps))

def find_optimal_threshold(gray_image):
    hist, _ = np.histogram(gray_image, bins=np.arange(257), density=True)
    cdf = hist.cumsum()
    max_entropy = -1
    optimal_threshold = 0
    
    for threshold in range(0, 1, 0.05):
        lower_part = hist[:threshold]
        upper_part = hist[threshold:]
        
        lower_prob = lower_part / lower_part.sum()
        upper_prob = upper_part / upper_part.sum()
        
        lower_entropy = calculate_entropy(lower_prob)
        upper_entropy = calculate_entropy(upper_prob)
        
        total_entropy = lower_entropy + upper_entropy
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_threshold = threshold
    
    return optimal_threshold

def deg_to_rad(deg):
    return deg * np.pi / 180

def rotation_matrix_x(angle_rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

def rotation_matrix_y(angle_rad):
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

def rotation_matrix_z(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

def calculate_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll_rad = deg_to_rad(roll_deg)
    pitch_rad = deg_to_rad(pitch_deg)
    yaw_rad = deg_to_rad(yaw_deg)
    
    Rx = rotation_matrix_x(roll_rad)
    Ry = rotation_matrix_y(pitch_rad)
    Rz = rotation_matrix_z(yaw_rad)
    
    R = Rz @ Ry @ Rx
    return R

def energy_mask(color: torch.Tensor, th_1=0.1, th_2=0.9):
    """
    mask out the background(black). set to 0 to mask black only, and other value(0, 1) to filter pixels with certain brightness
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=color.device).view(3, 1, 1)

    # Ensure color is in [0,1]. Some callers may pass uint8 images (0..255).
    # If the image max is > 2 we assume 0..255 and normalize.
    with torch.no_grad():
        cmax = float(color.max())
        if cmax > 2.0:
            color = color / 255.0

    gray = torch.sum(color * weights, dim=0).detach()  # mask should not have grad

    # Build boolean mask: True = foreground (keep)
    mask = (gray >= th_1) & (gray <= th_2)
    # Add leading batch/channel dim to match the pipeline expectation (1,H,W)
    zero_mask = mask[None]

    # Return a proper boolean tensor on the same device.
    return zero_mask.to(dtype=torch.bool, device=color.device)


def detect_collapse(color: torch.Tensor | None,
                    depth: torch.Tensor | None,
                    prev_area_frac: float | None = None,
                    area_drop_frac: float = 0.5,
                    min_area_frac: float = 0.05,
                    verbose: bool = False):
    """
    Heuristic collapse detector based on image and/or depth coverage.

    Args:
        color: [3,H,W] image tensor (can be None if only depth used)
        depth: [1,H,W] or [H,W] depth tensor (can be None if only color used)
        prev_area_frac: previous frame foreground fraction (optional)
        area_drop_frac: relative drop factor to declare collapse (e.g. 0.5 means current < prev*0.5)
        min_area_frac: absolute minimum area fraction to consider collapsed
        verbose: print debug info

    Returns:
        collapsed (bool), stats (dict) with keys 'area_frac' and 'depth_med'
    """
    H = W = None
    area_frac = None
    depth_med = None

    if depth is not None:
        d = depth.detach().cpu().numpy()
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        valid = (d > 0) & np.isfinite(d)
        area_frac = float(valid.mean())
        if valid.sum() > 0:
            depth_med = float(np.median(d[valid]))
        H, W = d.shape

    if color is not None and area_frac is None:
        c = color.detach().cpu().numpy()
        if c.ndim == 3:
            # expect CHW
            if c.shape[0] == 3:
                gray = c.mean(axis=0)
            else:
                gray = c
        else:
            gray = c
        valid = (gray > (gray.mean() * 0.5))
        area_frac = float(valid.mean())
        H, W = gray.shape

    if area_frac is None:
        # no signal available
        return False, {'area_frac': None, 'depth_med': None}

    collapsed = False
    if prev_area_frac is not None:
        if area_frac < prev_area_frac * area_drop_frac:
            collapsed = True
    if area_frac < min_area_frac:
        collapsed = True

    if verbose:
        print(f"detect_collapse: area_frac={area_frac:.4f} prev={prev_area_frac} depth_med={depth_med} -> collapsed={collapsed}")

    return collapsed, {'area_frac': area_frac, 'depth_med': depth_med}


def adaptive_collapse_densify(params, variables, curr_data, time_idx, mean_sq_dist_method,
                              config: dict | None = None, optimizer=None, prev_area_frac=None,
                              aggressive_cfg: dict | None = None,
                              use_grn=False, grn_model=None, gate_conf=None, verbose=False):
    """
    If a collapse is detected on the current frame `curr_data`, call `add_new_gaussians`
    with more aggressive parameters to densify the representation in missing/occluded regions.

    This is a helper wrapper that does not change global control flow; it returns
    the updated (params, variables) pair from `add_new_gaussians` so callers can use it
    inline during mapping.

    Args:
        params, variables: current scene state
        curr_data: dict with keys 'im', 'depth', 'intrinsics', 'w2c', 'cam' as in your pipeline
        time_idx: integer frame index
        mean_sq_dist_method: pass-through to add_new_gaussians
        optimizer: optional optimizer (not used here but kept for API symmetry)
        prev_area_frac: previous frame area fraction used by detect_collapse
        aggressive_cfg: dict overriding aggressive add_new_gaussians defaults
        use_grn, grn_model, gate_conf: passed to add_new_gaussians
    Returns:
        params_iter, variables, info
    """
    # Lazy import to avoid circular dependencies at module import time
    from utils.slam_helpers import add_new_gaussians

    # Default aggressive settings
    # Aggressive defaults tuned for collapsed-lumen densification: smaller init scale,
    # more bases to cover local complexity, and random initialization to avoid bias.
    # Derive defaults in the following precedence (highest -> lowest):
    # aggressive_cfg -> config (experiment) -> params-derived heuristic -> hardcoded fallback
    try:
        if params is not None and isinstance(params, dict) and 'deform_weights' in params:
            params_derived_nr_basis = int(params['deform_weights'].shape[1])
        else:
            params_derived_nr_basis = None
    except Exception:
        params_derived_nr_basis = None

    aggressive_cfg = aggressive_cfg or {}

    def _cfg_get(key, nested_keys=None, default=None):
        # check aggressive_cfg first
        if key in aggressive_cfg:
            return aggressive_cfg[key]
        # then check experiment config if provided
        if config is not None:
            # allow nested lookup via a list of keys
            if nested_keys is None:
                return config.get(key, default)
            else:
                cur = config
                for nk in nested_keys:
                    if isinstance(cur, dict) and nk in cur:
                        cur = cur[nk]
                    else:
                        cur = None
                        break
                if cur is not None:
                    return cur
        return default

    nr_basis = _cfg_get('nr_basis', nested_keys=['mapping', 'aggressive_nr_basis'], default=None)
    if nr_basis is None:
        # try known config locations
        nr_basis = _cfg_get('nr_basis', nested_keys=['deforms', 'nr_basis'], default=None)
    if nr_basis is None:
        nr_basis = params_derived_nr_basis or 8

    use_distributed_biases = _cfg_get('use_distributed_biases', nested_keys=['deforms', 'use_distributed_biases'], default=False)
    total_timescale = _cfg_get('total_timescale', nested_keys=['deforms', 'total_timescale'], default=None)
    use_grn_final = _cfg_get('use_grn', nested_keys=['GRN', 'use_grn'], default=use_grn)
    use_deform = _cfg_get('use_deform', nested_keys=['deforms', 'use_deformations'], default=True)
    deformation_type = _cfg_get('deformation_type', nested_keys=['deforms', 'deform_type'], default='gaussian')
    num_frames = _cfg_get('num_frames', default=_cfg_get('num_frames', nested_keys=['data', 'num_frames'], default=1))
    random_initialization = _cfg_get('random_initialization', nested_keys=['GRN', 'random_initialization'], default=True)
    init_scale = _cfg_get('init_scale', nested_keys=['GRN', 'init_scale'], default=0.02)
    cam = aggressive_cfg.get('cam', curr_data.get('cam', None))
    reduce_gaussians = _cfg_get('reduce_gaussians', nested_keys=['gaussian_reduction', 'reduce_gaussians'], default=False)
    reduction_type = _cfg_get('reduction_type', nested_keys=['gaussian_reduction', 'reduction_type'], default='random')
    reduction_fraction = _cfg_get('reduction_fraction', nested_keys=['gaussian_reduction', 'reduction_fraction'], default=0.5)
    gate_conf = gate_conf or _cfg_get('gate_conf', nested_keys=['deforms', 'gates'], default=gate_conf)

    aggressive_defaults = dict(
        nr_basis=int(nr_basis),
        use_distributed_biases=bool(use_distributed_biases),
        total_timescale=total_timescale,
        use_grn=use_grn_final,
        grn_model=grn_model,
        use_deform=bool(use_deform),
        deformation_type=deformation_type,
        num_frames=int(num_frames) if num_frames is not None else 1,
        random_initialization=bool(random_initialization),
        init_scale=float(init_scale),
        cam=cam,
        reduce_gaussians=bool(reduce_gaussians),
        reduction_type=reduction_type,
        reduction_fraction=float(reduction_fraction),
        gate_conf=gate_conf,
    )

    # Run detection using current data
    collapsed, stats = detect_collapse(curr_data.get('im', None), curr_data.get('depth', None), prev_area_frac=prev_area_frac)
    info = {'collapsed': collapsed, 'detector_stats': stats}

    if collapsed:
        if verbose:
            print(f"adaptive_collapse_densify: collapse detected at t={time_idx}, running aggressive add_new_gaussians")
        params_iter, variables = add_new_gaussians(params, variables, curr_data, sil_thres=0.5,
                                                   time_idx=time_idx, mean_sq_dist_method=mean_sq_dist_method,
                                                   use_simplification=True,
                                                   nr_basis=aggressive_defaults['nr_basis'],
                                                   use_distributed_biases=aggressive_defaults['use_distributed_biases'],
                                                   total_timescale=aggressive_defaults['total_timescale'],
                                                   use_grn=aggressive_defaults['use_grn'],
                                                   grn_model=aggressive_defaults['grn_model'],
                                                   use_deform=aggressive_defaults['use_deform'],
                                                   deformation_type=aggressive_defaults['deformation_type'],
                                                   num_frames=aggressive_defaults['num_frames'],
                                                   random_initialization=aggressive_defaults['random_initialization'],
                                                   init_scale=aggressive_defaults['init_scale'],
                                                   cam=aggressive_defaults['cam'],
                                                   reduce_gaussians=aggressive_defaults['reduce_gaussians'],
                                                   reduction_type=aggressive_defaults['reduction_type'],
                                                   reduction_fraction=aggressive_defaults['reduction_fraction'],
                                                   gate_conf=aggressive_defaults['gate_conf'])
        info['action'] = 'added_gaussians'
        return params_iter, variables, info

    info['action'] = 'none'
    return params, variables, info
