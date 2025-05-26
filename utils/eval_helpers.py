import cv2
from PIL import Image
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_external import build_rotation,calc_psnr
from utils.slam_helpers import transform_to_frame, transform_to_frame_eval, transformed_params2rendervar, transformed_params2depthplussilhouette,transformed_GRNparams2rendervar,transformed_GRNparams2depthplussilhouette,align_shift_and_scale


from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.time_helper import Timer
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error

# def align_shift_and_scale(gt_disp, pred_disp):

#     t_gt = np.median(gt_disp)
#     s_gt = np.mean(np.abs(gt_disp - t_gt))
    
#     t_pred = np.median(pred_disp)
#     s_pred = np.mean(np.abs(pred_disp - t_pred))
#     pred_disp_aligned = (pred_disp - t_pred) * (s_gt / s_pred) + t_gt

#     return pred_disp_aligned, t_gt, s_gt, t_pred, s_pred


# def align_shift_and_scale(gt_disp, pred_disp,mask):
#     ssum = torch.sum(mask, (1, 2))
#     valid = ssum > 0

    
#     t_gt = torch.median((gt_disp[valid]*mask[valid]).view(valid.sum(),-1),dim = 1).values
#     # print(t_gt)
#     # print(gt_disp[valid].view(valid.sum(),-1).shape,t_gt.shape)

#     s_gt = torch.mean(torch.abs(gt_disp[valid].view(valid.sum(),-1)- t_gt[:,None]),1)
#     t_pred = torch.median((pred_disp[valid]*mask[valid]).view(valid.sum(),-1),dim = 1).values
#     s_pred = torch.mean(torch.abs(pred_disp[valid].view(valid.sum(),-1)- t_pred[:,None]),1)
#     # print(pred_disp.view(gt_disp.shape[0],-1).shape,t_pred.shape,s_pred.shape)
#     pred_disp_aligned = (pred_disp.view(pred_disp.shape[0],-1)- t_pred[:,None])/s_pred[:,None]
#     pred_disp_aligned = pred_disp_aligned.view(pred_disp.shape[0],pred_disp.shape[1],pred_disp.shape[2])


#     gt_disp_aligned = (gt_disp.view(gt_disp.shape[0],-1)- t_gt[:,None])/s_gt[:,None]
#     gt_disp_aligned = gt_disp_aligned.view(gt_disp.shape[0],gt_disp.shape[1],gt_disp.shape[2])
#     return  gt_disp_aligned, pred_disp_aligned,t_gt, s_gt, t_pred, s_pred

def evaluate_ate(gt_traj, est_traj, plot_traj = False):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    rot, trans, trans_error = align(gt_traj_pts, est_traj_pts)

    gt_traj_aligned = rot* gt_traj_pts+trans
    gt_traj_aligned = np.array(gt_traj_aligned)
    fig = plt.figure()

# syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    x_gt = gt_traj_aligned[0,:]
    y_gt = gt_traj_aligned[1,:]
    z_gt = gt_traj_aligned[2,:]

    x_est = est_traj_pts[0,:]
    y_est = est_traj_pts[1,:]
    z_est = est_traj_pts[2,:]
    print(x_gt[0])
    ax.plot3D(x_gt, y_gt, z_gt, 'green', label='Ground Truth Trajectory')
    ax.plot3D(x_est, y_est, z_est, 'red', label='Estimated Trajectory')
    plt.show()

    avg_trans_error = trans_error.mean()

    return avg_trans_error
        

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, 
                         save_plot=False, diff_rgb=None):
    
    if depth.max() > 1.0:
        depth = depth / 100.0 * 2.55
    if rastered_depth.max() > 1.0:
        rastered_depth = rastered_depth / 100.0 * 2.55
    
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb.cpu(), cmap='jet', vmin=0, vmax=6)
        axs[0, 2].set_title("Diff RGB L1")
    else:
        axs[0, 2].imshow(presence_sil_mask, cmap='gray')
        axs[0, 2].set_title("Rasterized Silhouette")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")
    for ax in axs.flatten():
        ax.axis('off')
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, online_time_idx=None,
                    global_logging=True):
    if i % every_i == 0 or i == 1:
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                     transformed_pts)
        depth_sil, _, _ = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0) & (data['depth'] < 1e10)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _ = Renderer(raster_settings=data['cam'])(**rendervar)
        if tracking:
            psnr = calc_psnr(im * presence_sil_mask , data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        

# def deform_gaussians(params,time,deform_grad):

#     if deform_grad:
#         weights = params['deform_weights']
#         stds = params['deform_stds']
#         biases = params['deform_biases']
#     else:
#         weights = params['deform_weights'].detach()
#         stds = params['deform_stds'].detach()
#         biases = params['deform_biases'].detach()

#     deform = torch.sum(weights*torch.exp(-1/(2*stds**2)*(time-biases)**2),1) # Nx10 gaussians deformations
#     deform_xyz = deform[:,:3]
#     deform_rots = deform[:,3:7]
#     deform_scales = deform[:,7:10]

#     xyz = params['means3D']+deform_xyz
#     rots = params['unnorm_rotations']+deform_rots
#     scales = params['log_scales']+deform_scales

#     return xyz,rots,scales


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
        xyz = params['means3D'][time]
        rots = params['unnorm_rotations'][time]
        scales = params['log_scales'][time]
        opacities = params['logit_opacities'][time]
        colors = params['rgb_colors'][time]

    return xyz, rots, scales,opacities, colors


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    psnr = 20*np.log10(np.max(gt)/np.sqrt(rmse))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, psnr

def eval_save(dataset, final_params, eval_dir, sil_thres, 
         mapping_iters, add_new_gaussians, save_renders=True,use_grn = True):
    # timer = Timer()
    # timer.start()
    split = False
    if type(dataset) is list:
        assert dataset[2] == 'C3VD', 'This eval is only for data splits with C3VD(7::8). Note it.'
        train_dataset, eval_dataset = dataset[:2]
        split = True
        num_frames = len(train_dataset) + len(eval_dataset)
    else:
        num_frames = len(dataset)
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    depth_errors_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_renders:
        render_rgb_dir = os.path.join(eval_dir, "color")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(render_depth_dir, exist_ok=True)

    gt_w2c_list = []
    gt_position_list = []
    est_position_list = []
    
    def get_idx(total_time_idx):
        if split:
            if (total_time_idx + 1) % 8 != 0:
                dataset_type = 'train'
                _dataset = train_dataset
                time_idx = total_time_idx - (total_time_idx // 8)
            else:
                dataset_type = 'test'
                _dataset = eval_dataset
                time_idx = total_time_idx//8
        else:
            dataset_type = 'train'
            _dataset = dataset
            time_idx = total_time_idx
        return dataset_type, _dataset, time_idx
    
    for total_time_idx in tqdm(range(num_frames)):
        dataset_type, dataset, time_idx = get_idx(total_time_idx)
        gt_w2c_list.append(torch.linalg.inv(dataset.get_pose(time_idx).cpu()))
        gt_position_list.append(gt_w2c_list[-1][:3, 3])
        if total_time_idx == 0:
            est_position_list.append(gt_position_list[0])
        elif dataset_type == 'train':
            interm_cam_trans = final_params['cam_trans'][..., time_idx].detach().cpu()
            est_position_list.append(interm_cam_trans.squeeze())
            
    if split:
        all_idx = set(range(num_frames))
        train_idx = None
        eval_idx = set(range(7, num_frames, 8)) # we don't want to expect eval in early frames
        train_idx = all_idx - eval_idx
        eval_idx = sorted(list(eval_idx))
        train_idx = sorted(list(train_idx))
        # horn gt w2c for nvs render. Nothing to do with metrics like ate
        rot, trans, _ = align(torch.stack(gt_position_list)[train_idx].detach().cpu().numpy().T, torch.stack(est_position_list).detach().cpu().numpy().T)
        horn_gt_position = [rot @ gt_position.numpy()+trans.squeeze() for gt_position in gt_position_list]
        
    for total_time_idx in tqdm(range(num_frames)):
        dataset_type, dataset, time_idx = get_idx(total_time_idx)
        # print(time_idx)
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        
        # if torch.max(depth) > 1.0:
        #     depth = depth / 100.0 # only for c3vd. This scale is exact.
        
        if total_time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

        # Get current frame Gaussians
        local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(final_params,total_time_idx,False,deformation_type = 'simple')
        if dataset_type == 'train':
            cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., time_idx].detach())
            cam_tran = final_params['cam_trans'][..., time_idx].detach()
            transformed_pts = transform_to_frame_eval(final_params,local_means, (cam_rot, cam_tran))
        else:
            w2c = gt_w2c_list[total_time_idx]
            # w2c[:3, :3] = torch.Tensor(rot) @ w2c[:3, :3]
            w2c[:3, 3] = torch.Tensor(horn_gt_position[total_time_idx])
            transformed_pts = transform_to_frame_eval(final_params,local_means, rel_w2c=w2c.cuda()) # use gt pose to render
            
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        # visall = False # NOTE: for debug
        # if not visall and dataset_type == 'train':
        #     continue
        if time_idx in dataset.eval_idx:
            # # Initialize Render Variables
            # rendervar = transformed_params2rendervar(final_params, transformed_pts,local_rots,local_scales)
            # depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
            #                                                              transformed_pts,local_rots,local_scales)
            if use_grn:
                rendervar = transformed_GRNparams2rendervar(final_params, transformed_pts,local_rots,local_scales,local_opacities,local_colors)

                depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(final_params, curr_data['w2c'],
                                                                            transformed_pts,local_rots,local_scales,local_opacities)
            else:
                rendervar = transformed_params2rendervar(final_params, transformed_pts,local_rots,local_scales,local_opacities,local_colors)

                depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                            transformed_pts,local_rots,local_scales,local_opacities)

            # Render Depth & Silhouette
            depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
            rastered_depth = depth_sil[0, :, :].unsqueeze(0)
            
            # if rastered_depth.max() > 1.0:
            #     rastered_depth = rastered_depth / 100.0
            
            # Mask invalid depth in GT
            valid_depth_mask = (curr_data['depth'] > 0) & (curr_data['depth'] < 1e10)
            rastered_depth_viz = rastered_depth.detach()
            rastered_depth = rastered_depth * valid_depth_mask
            silhouette = depth_sil[1, :, :]
            presence_sil_mask = (silhouette > sil_thres)
            
            
            # Render RGB and Calculate PSNR
            # timer.lap('rest part', stage=0)
            # for _ in range(100):
            #     im, _, _= Renderer(raster_settings=curr_data['cam'])(**rendervar)
            # timer.lap('render part', stage=1)
            im, _, _= Renderer(raster_settings=curr_data['cam'])(**rendervar)
            # if mapping_iters==0 and not add_new_gaussians:
            #     weighted_im = im * presence_sil_mask * valid_depth_mask
            #     weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
            # else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                            data_range=1.0, size_average=True)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                        torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)

            # Compute Depth RMSE

            # gt_depth_aligned,rastered_depth_aligned,t_gt,s_gt,t_pred,s_pred = align_shift_and_scale( curr_data['depth'].cpu().detach(),rastered_depth_viz.cpu().detach())
            # gt_depth_aligned[gt_depth_aligned<1e-3] = 1e-3
            # rastered_depth_aligned[rastered_depth_aligned<1e-3] = 1e-3
            # gt_depth_aligned[gt_depth_aligned>150] = 150
            # rastered_depth_aligned[rastered_depth_aligned>150] = 150
            gt_depth_aligned = curr_data['depth']
            _,_,t_gt,s_gt,t_pred,s_pred = align_shift_and_scale(curr_data['depth'], rastered_depth_viz,valid_depth_mask)
            rastered_depth_aligned = (rastered_depth_viz-t_pred)*(s_gt/s_pred)+t_gt

            # fig,ax = plt.subplots(2,3)
            # im0 = ax[0,0].imshow(gt_depth_aligned.squeeze().cpu().detach())
            # ax[0,0].set_title("GT Depth aligned, s_gt: {:.2f}, t_gt: {:.2f}".format(s_gt.item(),t_gt.item()))
            # plt.colorbar(im0,ax=ax[0,0])
            # im1 = ax[0,1].imshow(rastered_depth_aligned.squeeze().cpu().detach())
            # ax[0,1].set_title("Rastered Depth aligned, s_pred: {:.2f}, t_pred: {:.2f}".format(s_pred.item(),t_pred.item()))
            # plt.colorbar(im1,ax=ax[0,1])
            # im2 = ax[0,2].imshow(np.abs(gt_depth_aligned.squeeze().cpu().detach()-rastered_depth_aligned.squeeze().cpu().detach()*valid_depth_mask.squeeze().cpu().detach()))
            # plt.colorbar(im2,ax=ax[0,2])


            # im3 = ax[1,0].imshow(curr_data['depth'].squeeze().cpu().detach())
            # ax[1,0].set_title("GT Depth (not aligned)")
            # plt.colorbar(im3,ax=ax[1,0])
            # im4 = ax[1,1].imshow(rastered_depth_viz.squeeze().cpu().detach())
            # ax[1,1].set_title("Rastered Depth (not aligned)")
            # plt.colorbar(im4,ax=ax[1,1])
            # im5 = ax[1,2].imshow(valid_depth_mask.squeeze().cpu().detach())
            # ax[1,2].set_title("Valid Depth Mask")
            # plt.colorbar(im5,ax=ax[1,2])
            # plt.show()


            # gt_depth_aligned = curr_data['depth'][valid_depth_mask].cpu().detach().numpy()
            # rastered_depth_aligned = rastered_depth_viz[valid_depth_mask].cpu().detach().numpy()
            # gt_depth_aligned[gt_depth_aligned<1e-3] = 1e-3
            # rastered_depth_aligned[rastered_depth_aligned<1e-3] = 1e-3
            # gt_depth_aligned[gt_depth_aligned>150] = 150
            # rastered_depth_aligned[rastered_depth_aligned>150] = 150

            # rastered_depth_aligned,t_gt,s_gt,t_pred,s_pred = align_shift_and_scale( gt_depth_aligned,rastered_depth_aligned)



            print("s_gt: {:.2f}, t_gt: {:.2f}, s_pred: {:.2f}, t_pred: {:.2f}".format(s_gt.item(),t_gt.item(),s_pred.item(),t_pred.item()))

            error = compute_errors((gt_depth_aligned[valid_depth_mask].cpu().detach().numpy()), (rastered_depth_aligned[valid_depth_mask].cpu().detach().numpy()))
            depth_errors_list.append(error)
            print("Abs Rel: {:.4f}, Sq Rel: {:.4f}, RMSE: {:.4f}, RMSE Log: {:.4f}, A1: {:.4f}, A2: {:.4f}, A3: {:.4f}, PSNR: {:.4f}".format(*error))

            diff_depth_rmse = torch.sqrt((((rastered_depth_aligned - gt_depth_aligned)) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth_aligned - gt_depth_aligned))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
            rmse_list.append(rmse.cpu().numpy())
            l1_list.append(depth_l1.cpu().numpy())

            if save_renders:
                # Save Rendered RGB and Depth
                viz_render_im = torch.clamp(im, 0, 1)
                viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
                vmin = 0
                vmax = 6
                viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
                normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
                # depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
                save_im = np.clip(viz_render_im*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(render_rgb_dir, "color_{:04d}.png".format(total_time_idx)), cv2.cvtColor(save_im, cv2.COLOR_RGB2BGR))
                # Image.fromarray(np.uint8(viz_render_depth*10/2.55), 'L').save(os.path.join(render_depth_dir, "depth_{:04d}.tiff".format(total_time_idx)))
                save_depth = np.clip(viz_render_depth*655.35, 0, 65535)
                Image.fromarray(np.uint16(save_depth)).save(os.path.join(render_depth_dir, "depth_{:04d}.tiff".format(total_time_idx)))
                # cv2.imwrite(os.path.join(render_depth_dir, "depth_{:04d}.tiff".format(total_time_idx)), viz_render_depth*100)
            
            # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
            fig_title = "Time Step: {}".format(total_time_idx)
            plot_name = "%04d" % total_time_idx
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                    psnr, depth_l1, fig_title, plot_dir, 
                                    plot_name=plot_name, save_plot=True)

    # Get the final camera trajectory
    num_frames = final_params['cam_unnorm_rots'].shape[-1]
    latest_est_w2c = first_frame_w2c
    latest_est_w2c_list = []
    latest_est_w2c_list.append(latest_est_w2c)
    valid_gt_w2c_list = []
    valid_gt_w2c_list.append(gt_w2c_list[0])
    for idx in range(1, num_frames):
        interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
        interm_cam_trans = final_params['cam_trans'][..., idx].detach()
        intermrel_w2c = torch.eye(4).cuda().float()
        intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
        intermrel_w2c[:3, 3] = interm_cam_trans
        latest_est_w2c = intermrel_w2c
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list.append(gt_w2c_list[idx])
    
    # save poses for ate eval
    # if split:
    with open(os.path.join(eval_dir, 'gt_train_w2c.txt'), 'w') as f:
        for tensor in [gt_w2c_list[idx] for idx in range(num_frames)]:
            line = ''
            for v in tensor.reshape(-1).numpy():
                line += str(v) + ','
            f.write(line[:-1] + '\n')
    with open(os.path.join(eval_dir, 'est_w2c.txt'), 'w') as f:
        for tensor in latest_est_w2c_list:
            line = ''
            for v in tensor.reshape(-1).cpu().numpy():
                line += str(v) + ','
            f.write(line[:-1] + '\n')
    # timer.end()
    # gt_w2c_list = valid_gt_w2c_list

    # # Calculate ATE RMSE
    ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
    print("Final Average ATE RMSE: {:.2f} mm".format(ate_rmse*100))
    depth_error_metrics = ['Abs Rel', 'Sq Rel', 'RMSE', 'RMSE Log', 'A1', 'A2', 'A3', 'PSNR']
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    depth_errors_np = {key: [] for key in depth_error_metrics}
    for error in depth_errors_list:
        for id,metric in enumerate(error):
            depth_errors_np[depth_error_metrics[id]].append(metric)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_depth_errors = {key: np.mean(depth_errors_np[key]) for key in depth_errors_np.keys()}
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} mm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} mm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average Depth Errors: {}".format(avg_depth_errors))

    # # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    # np.savetxt(os.path.join(eval_dir, 'depth_errors.txt'), depth_errors_np)
    np.savetxt(os.path.join(eval_dir, 'depth_errors.txt'),np.array([depth_errors_np[key] for key in depth_errors_np.keys()]).T,header = '       '.join(depth_errors_np.keys()), fmt='%.6f',delimiter=', ')


    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (mm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} mm, ATE RMSE: {:.2f} mm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    plt.close()
