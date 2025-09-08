import torch
import lpips
import os
import glob
import cv2
import numpy as np
from pytorch_msssim import ms_ssim
from PIL import Image

lpips_model = lpips.LPIPS(net='alex')
    
def lsFile(folder_path, ext='png'):
    """
    bash like ls command to list files in a folder with a specific extension.
    """
    search_pattern = os.path.join(folder_path, '*.'+ext)
    files = glob.glob(search_pattern)
    sorted_files = sorted(files)
    return sorted_files


def read_pose_file(pose_file):
    """
    Reads a pose file and extracts camera poses. Each pose is expected to be in a comma-separated format, representing a 4x4 transformation matrix.

    Args:
        pose_file: The file path to the pose file. The file should contain lines, each line representing a camera pose as 16 comma-separated floats that can be reshaped into a 4x4 matrix.

    Returns:
        poses: A list of 4x4 numpy arrays, each array representing a camera pose as extracted from the file.
    """
    
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        poses = [np.array([float(x) for x in line.split(',')]).reshape(4, 4) for line in lines]
        if poses[-1][:3, 3].sum() == 0:
            poses = [pose.T for pose in poses]
    return poses


def calculate_psnr(img1, img2):
    """Calculates the PSNR between two images.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The PSNR between the two images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Avoid division by zero
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def calculate_ssim(img1, img2):
    """
    Calculates the MS-SSIM between two images using PyTorch.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The MS-SSIM between the two images.
    """
    if np.max(img1) > 1:
        img1 = img1/255.0
        img2 = img2/255.0
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    ms_ssim_value = ms_ssim(img1_tensor, img2_tensor, data_range=1.0)
    return ms_ssim_value


def calculate_lpips(img1, img2):
    """Calculates the LPIPS between two images.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The LPIPS between the two images.
    """
    if np.max(img1) > 1:
        img1 = img1/255.0
        img2 = img2/255.0
        
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        lpips_distance = lpips_model(img1_tensor, img2_tensor)
    return lpips_distance.item()


def calculate_depth_rmse(depth1, depth2):
    """
    Calculates the RMSE (Root Mean Square Error) between two depth maps.

    Args:
        depth1: The first depth map as a NumPy array.
        depth2: The second depth map as a NumPy array.

    Returns:
        The RMSE value between the two depth maps.
    """
    mse = np.mean((depth1 - depth2) ** 2)
    rmse = np.sqrt(mse)
    return rmse


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
    #np.set_printoptions(precision=3, suppress=True)
    # --- ensure both trajectories have the same number of columns ---
    # (this fixes the IndexError when one has e.g. 514 points and the other 513)
    n = min(model.shape[1], data.shape[1])
    model = model[:, :n]
    data  = data[:, :n]
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1] - 1):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return model_aligned, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = np.array(gt_traj_pts).T
    est_traj_pts = np.array(est_traj_pts).T

    gt_aligned, trans_error = align(gt_traj_pts, est_traj_pts)
    avg_trans_error = trans_error.mean()

    return avg_trans_error, gt_aligned, est_traj_pts
        
        
def rgb_metrics(gt, render):
    """
    Calculates the PSNR, MS-SSIM, and LPIPS metrics for RGB images.
    """
    color_gt = os.path.join(gt, 'color')
    color_render = os.path.join(render, 'color')
    color_files1 = lsFile(color_gt)[7::8]
    color_files2 = lsFile(color_render)
    print(color_files2)
    if '0000' in os.path.basename(color_files2[0]):
        color_files2 = color_files2[7::8]

    # load and convert to RGB
    color1 = [
        cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32), cv2.COLOR_BGR2RGB)
        for p in color_files1
    ]
    color2 = [
        cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32), cv2.COLOR_BGR2RGB)
        for p in color_files2
    ]

    # resize each render image to match its GT counterpart
    for i in range(len(color2)):
        h_gt, w_gt = color1[i].shape[:2]
        if color2[i].shape[:2] != (h_gt, w_gt):
            color2[i] = cv2.resize(color2[i], (w_gt, h_gt), interpolation=cv2.INTER_AREA)

    psnr_list  = [calculate_psnr(color1[i], color2[i]) for i in range(len(color1))]
    ssim_list  = [calculate_ssim(color1[i], color2[i]) for i in range(len(color1))]
    lpips_list = [calculate_lpips(color1[i], color2[i]) for i in range(len(color1))]

    mean_psnr  = np.mean(psnr_list)
    mean_ssim  = np.mean(ssim_list)
    mean_lpips = np.mean(lpips_list)

    return mean_psnr, mean_ssim, mean_lpips, psnr_list, ssim_list, lpips_list


def depth_metrics(gt, render):
    """
    Calculates the RMSE for depth maps.
    """
    depth_gt     = os.path.join(gt, 'depth')
    depth_render = os.path.join(render, 'depth')
    depth_files1 = lsFile(depth_gt, 'tiff')[7::8]
    depth_files2 = lsFile(depth_render, 'tiff')
    if '0000' in os.path.basename(depth_files2[0]):
        depth_files2 = depth_files2[7::8]

    # load and normalize
    depth1 = [np.array(Image.open(p)).astype(np.float32) / 2.55   for p in depth_files1]
    depth2 = [np.array(Image.open(p)).astype(np.float32) / 655.35 for p in depth_files2]
    depth2 = [np.clip(d, 0, 100) for d in depth2]

    # resize each render depth to match its GT counterpart
    for i in range(len(depth2)):
        h_gt, w_gt = depth1[i].shape
        if depth2[i].shape != (h_gt, w_gt):
            depth2[i] = cv2.resize(depth2[i], (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)

    rmse_list = [calculate_depth_rmse(depth1[i], depth2[i]) for i in range(len(depth1))]
    mean_rmse = np.mean(rmse_list)

    return mean_rmse, rmse_list


def pose_metrics(gt_w2c_path, est_w2c_path, align_gt_path=None):
    """
    Calculates the Average Trajectory Error (ATE) between ground truth and estimated camera poses.
    """
    gt_w2c  = read_pose_file(gt_w2c_path)
    est_w2c = read_pose_file(est_w2c_path)
    if align_gt_path is not None:
        gt_w2c = read_pose_file(align_gt_path)

    ate, est, gt = evaluate_ate(est_w2c, gt_w2c)
    return ate, np.array(gt), np.array(est)
