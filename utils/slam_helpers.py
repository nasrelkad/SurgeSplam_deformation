import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import energy_mask
import torchvision
import numpy as np
import cv2


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
        'rotations': F.normalize(local_rots),
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


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
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
        'rotations': F.normalize(local_rots),
        'opacities': torch.sigmoid(local_opacities),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
    }
    if local_scales.shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(local_scales, (1, 3)))
    else:
        rendervar['scales'] = torch.exp(local_scales)
    return rendervar


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


def transform_to_frame_eval(params, local_means,camrt=None, rel_w2c=None):
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
    if rel_w2c is None:
        cam_rot, cam_tran = camrt
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    pts = local_means.detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

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
        'rotations': F.normalize(local_rots),
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
        'rotations': F.normalize(local_rots),
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




def deform_gaussians(params, time, deform_grad, N=5,deformation_type = 'gaussian'):
    """
    Calculate deformations using the N closest basis functions based on |time - bias|.

    Args:
        params (dict): Dictionary containing deformation parameters.
        time (torch.Tensor): Current time step.
        deform_grad (bool): Whether to calculate gradients for deformations.
        N (int): Number of closest basis functions to consider.

    Returns:
        xyz (torch.Tensor): Updated 3D positions.
        rots (torch.Tensor): Updated rotations.
        scales (torch.Tensor): Updated scales.
    """
    if deformation_type =='gaussian':
        if True:
            if deform_grad:
                weights = params['deform_weights']
                stds = params['deform_stds']
                biases = params['deform_biases']
            else:
                weights = params['deform_weights'].detach()
                stds = params['deform_stds'].detach()
                biases = params['deform_biases'].detach()

            # Calculate the absolute difference between time and biases
            time_diff = torch.abs(time - biases)

            # Get the indices of the N smallest time differences
            _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

            # Create a mask to select only the top N basis functions
            mask = torch.zeros_like(time_diff, dtype=torch.float)
            mask.scatter_(1, top_indices, 1.0)

            # Apply the mask to weights and biases
            masked_weights = weights * mask
            masked_biases = biases * mask

            # Calculate deformations
            deform = torch.sum(
                masked_weights * torch.exp(-1 / (2 * stds**2) * (time - masked_biases)**2), dim=1
            )  # Nx10 gaussians deformations

            deform_xyz = deform[:, :3]
            deform_rots = deform[:, 3:7]
            deform_scales = deform[:, 7:10]
        else:
            if deform_grad:
                weights = params['deform_weights']
                stds = params['deform_stds']
                biases = params['deform_biases']
            else:
                weights = params['deform_weights'].detach()
                stds = params['deform_stds'].detach()
                biases = params['deform_biases'].detach()

            # Calculate the absolute difference between time and biases
            time_diff = torch.abs(time - biases)

            # Get the indices of the N smallest time differences
            _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

            # Create a mask to select only the top N basis functions
            mask = torch.zeros_like(time_diff, dtype=torch.float)
            mask.scatter_(1, top_indices, 1.0).detach()

            # Register a gradient hook to zero out gradients for irrelevant basis functions
            if deform_grad:
                def zero_out_irrelevant_gradients(grad):
                    return grad * mask

                weights.register_hook(zero_out_irrelevant_gradients)
                biases.register_hook(zero_out_irrelevant_gradients)
                stds.register_hook(zero_out_irrelevant_gradients)

            # Calculate deformations
            deform = torch.sum(
                weights * torch.exp(-1 / (2 * stds**2) * (time - biases)**2), dim=1
            )  # Nx10 gaussians deformations

            deform_xyz = deform[:, :3]
            deform_rots = deform[:, 3:7]
            deform_scales = deform[:, 7:10]

        xyz = params['means3D'] + deform_xyz
        rots = params['unnorm_rotations'] + deform_rots
        scales = params['log_scales'] + deform_scales
        opacities = params['logit_opacities']
        colors = params['rgb_colors']


    elif deformation_type == 'simple':
        # with torch.no_grad():
        xyz = params['means3D']
        rots = params['unnorm_rotations']
        scales = params['log_scales']
        opacities = params['logit_opacities']
        colors = params['rgb_colors']

    return xyz, rots, scales,opacities, colors



def initialize_deformations(params,nr_basis,use_distributed_biases,total_timescale= None):
    """Function to initialize deformation parameters

    Args:
        params (dict): dict containing all the gaussian parameters
        nr_basis (int): nr of basis functions to generate for each gaussian attribute
        total_timescale (int): the total timescale of the deformation model, used to evenly distribute biases across timespan

    Returns:
        params (dict): Updated params dict with the gaussian deformation parameters
    """
    # Means3D, unnorm rotations and log_scales should receive deformation params
    N = params['means3D'].shape[0]
    weights = torch.randn([N,nr_basis,10],requires_grad = True,device = 'cuda')*0.0 # We have N x nr_basis x 10 (xyz,scales,rots) weights
    stds = torch.ones([N,nr_basis,10],requires_grad = True,device = 'cuda')/0.1 # We have N x nr_basis x 10 (xyz,scales,rots) weights
    if not use_distributed_biases:
        biases = torch.randn([N,nr_basis,10],requires_grad = True,device = 'cuda')*0.0 # We have N x nr_basis x 10 (xyz,scales,rots) weights
    
    
    else:
        interval = torch.ceil(torch.tensor(total_timescale/nr_basis)) # For biases, we want to evenly distribute them across the timespan, so we have nr_basis biases at an interval of total_timescale/nr_basis
        arange = torch.arange(0,total_timescale,interval).unsqueeze(0).unsqueeze(-1)
        biases = torch.tile(arange,(N,1,10)) # Repeat the distributed biases to generate basis functions for each parameter

    params['deform_weights'] = weights
    params['deform_stds'] = stds
    params['deform_biases'] = biases


    return params


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
    if use_deform and deform_type == 'gaussian':
        params = initialize_deformations(params,nr_basis = nr_basis,use_distributed_biases=use_distributed_biases,total_timescale = total_timescale)
    # elif use_deform and deform_type == 'simple':
    #     params = initialize_simple_deformations(params,num_frames)
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
    # print(output)
    cols = torch.permute(output[0], (1, 2, 0)).reshape(-1, 8) # (C, H, W) -> (H, W, C) -> (H * W, C)
    

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
        num_pts = params_iter['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    else:
        params_iter = params
    return params_iter, variables

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