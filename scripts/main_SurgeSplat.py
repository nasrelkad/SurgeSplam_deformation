import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from datasets.gradslam_datasets import (
    load_dataset_config,
    EndoSLAMDataset,
    C3VDDataset,
    ScaredDataset,
    EndoNerfDataset,
    RARPDataset,
    HamlynDataset,
    StereoMisDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save
from utils.keyframe_selection import keyframe_selection_overlap, keyframe_selection_distance
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion,
    transformed_GRNparams2depthplussilhouette,transformed_GRNparams2rendervar,
    add_new_gaussians,grn_initialization,deform_gaussians,initialize_deformations,initialize_new_params,grn_initialization,
    get_mask,align_shift_and_scale, initialize_cv_deformations, initialize_xyzt, xyzt_time_gate, apply_xyzt_gate
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, remove_points
from utils.vis_utils import plot_video
from utils.time_helper import Timer

from models.SurgeDepth.dpt import SurgeDepth
import torchvision

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

import matplotlib.pyplot as plt

from PIL import Image
from typing import Dict, Optional
import math


def _ensure_motion_state(variables: Dict) -> Dict:
    state = variables.get('motion_state')
    if state is None:
        state = {
            'prev_tran': None,
            'pause_counter': 0,
            'is_contraction': False,
            'delta': 0.0,
            'last_step': 0.0,
            'forward_dir': None
        }
        variables['motion_state'] = state
    return state


def _compute_gate_thresh(base_gate: float, motion_cfg: Dict, motion_state: Dict) -> float:
    if not motion_cfg.get('enable', False):
        return base_gate
    gate = base_gate
    if motion_state.get('is_contraction'):
        gate += motion_cfg.get('gate_thresh_boost', 0.0)
    else:
        gate += motion_cfg.get('moving_gate_offset', 0.0)
    gate_cap = motion_cfg.get('gate_thresh_cap', 0.99)
    return float(max(0.0, min(gate, gate_cap)))


def _update_motion_state(variables: Dict, params_iter, time_idx: int, motion_cfg: Dict) -> Dict:
    state = _ensure_motion_state(variables)
    curr = params_iter['cam_trans'][..., time_idx].detach().view(-1).float().cpu()
    prev = state.get('prev_tran')

    delta = 0.0
    direction_min = motion_cfg.get('direction_min_delta', motion_cfg.get('contraction_speed_thresh', 1e-3))
    contraction_thresh = motion_cfg.get('contraction_speed_thresh', 1e-3)
    pause_frames = motion_cfg.get('min_pause_frames', 2)

    if prev is not None and prev.shape == curr.shape:
        delta_vec = curr - prev
        delta = torch.norm(delta_vec).item()
        if delta > direction_min:
            state['forward_dir'] = (delta_vec / (delta + 1e-8)).clone()
            state['last_step'] = delta
        else:
            state['last_step'] = state.get('last_step', 0.0) * motion_cfg.get('pause_forward_decay', 0.5)
    else:
        state['last_step'] = state.get('last_step', 0.0)

    if delta < contraction_thresh:
        state['pause_counter'] = state.get('pause_counter', 0) + 1
    else:
        state['pause_counter'] = 0
    state['is_contraction'] = state['pause_counter'] >= pause_frames
    state['delta'] = delta
    state['prev_tran'] = curr
    return state


def _update_scene_state(variables, motion_state, added_this_frame, missing_ratio=None, cfg=None):
    """
    Decide if we stay in 'protect' (no pruning) or go to 'slam' (normal).
    """
    state = variables.get("scene_state")
    if state is None:
        state = {"mode": "protect", "open_frames": 0, "recent_big_adds": 0}
        variables["scene_state"] = state

    mode = state.get("mode", "protect")

    motion_cfg = cfg.get('motion', {}) if cfg is not None else {}

    # thresholds – now partly configurable
    min_forward_speed = motion_cfg.get('min_forward_speed', 1e-3)        # if cam moves more than this → counts as forward
    needed_open_frames = motion_cfg.get('protect_open_frames', 15)       # how many good frames in a row to switch
    big_add_thresh = motion_cfg.get('big_add_thresh', 1500)              # if we added > this, scene is still revealing things
    max_recent_big_adds = 3         # allow a few big-add frames before switching
    openness_needed = 0.25          # if we pass missing_ratio, we can count it as open

    cam_is_moving = motion_state.get("delta", 0.0) > min_forward_speed
    still_adding_a_lot = added_this_frame is not None and added_this_frame > big_add_thresh
    scene_filled_enough = (missing_ratio is not None and missing_ratio < (1.0 - openness_needed))

    if mode == "protect":
        if still_adding_a_lot:
            # reset counter – we are still discovering tissue
            state["recent_big_adds"] = min(state["recent_big_adds"] + 1, max_recent_big_adds)
            state["open_frames"] = 0
        else:
            # no big add this frame → maybe scene is opening
            state["recent_big_adds"] = max(state["recent_big_adds"] - 1, 0)
            if cam_is_moving or scene_filled_enough:
                state["open_frames"] += 1
            else:
                # paused and still not adding – just keep waiting
                state["open_frames"] = max(state["open_frames"] - 1, 0)

        # switch condition: a few consecutive frames with no big adds AND camera moving
        if state["open_frames"] >= needed_open_frames and state["recent_big_adds"] == 0:
            state["mode"] = "slam"
            state["recent_big_adds"] = 0

    else:  # mode == "slam"
        # optional: if we suddenly start adding tons again (new big collapse),
        # we can fall back to protect
        if still_adding_a_lot and not cam_is_moving:
            state["mode"] = "protect"
            state["open_frames"] = 0
            state["recent_big_adds"] = 0

    return state


def _prob_to_logit(prob: float, eps: float = 1e-4) -> float:
    prob = max(min(prob, 1.0 - eps), eps)
    return math.log(prob / (1.0 - prob))


def _clamp_gaussian_params(params: Dict, maint_cfg: Dict) -> None:
    if not maint_cfg.get('enable', False):
        return
    with torch.no_grad():
        log_scales = params.get('log_scales')
        if log_scales is not None:
            clamp_min = maint_cfg.get('log_scale_min', None)
            clamp_max = maint_cfg.get('log_scale_max', None)
            if clamp_min is not None or clamp_max is not None:
                clamp_min = float(clamp_min) if clamp_min is not None else None
                clamp_max = float(clamp_max) if clamp_max is not None else None
                if clamp_min is not None and clamp_max is not None:
                    log_scales.data.clamp_(min=clamp_min, max=clamp_max)
                elif clamp_min is not None:
                    log_scales.data.clamp_(min=clamp_min)
                elif clamp_max is not None:
                    log_scales.data.clamp_(max=clamp_max)
        logit_opacities = params.get('logit_opacities')
        if logit_opacities is not None:
            clamp_min = maint_cfg.get('logit_min', None)
            clamp_max = maint_cfg.get('logit_max', None)
            if clamp_min is not None or clamp_max is not None:
                clamp_min = float(clamp_min) if clamp_min is not None else None
                clamp_max = float(clamp_max) if clamp_max is not None else None
                if clamp_min is not None and clamp_max is not None:
                    logit_opacities.data.clamp_(min=clamp_min, max=clamp_max)
                elif clamp_min is not None:
                    logit_opacities.data.clamp_(min=clamp_min)
                elif clamp_max is not None:
                    logit_opacities.data.clamp_(max=clamp_max)


def _update_opacity_and_visibility(params: Dict, variables: Dict, frame_idx: int,
                                   maint_cfg: Dict, active_mask: Optional[torch.Tensor]) -> None:
    if not maint_cfg.get('enable', False):
        return
    if 'logit_opacities' not in params or 'last_seen' not in variables:
        return

    logits = params['logit_opacities']
    device = logits.device
    last_seen = variables['last_seen']
    visibility_hits = variables['visibility_hits']
    frame_tensor = torch.tensor(float(frame_idx), device=device, dtype=last_seen.dtype)

    with torch.no_grad():
        decay_rate = float(maint_cfg.get('opacity_decay', 0.0))
        if decay_rate > 0.0:
            if last_seen.shape[0] != logits.shape[0]:
                # Geometry changed elsewhere; postpone maintenance until arrays are realigned.
                return
            dt = torch.where(last_seen >= 0.0, frame_tensor - last_seen, torch.zeros_like(last_seen))
            dt = torch.clamp(dt, min=0.0)
            decay = (dt * decay_rate).unsqueeze(-1) if logits.dim() > 1 else dt * decay_rate
            logits.data -= decay

        floor_prob = maint_cfg.get('opacity_floor', None)
        if floor_prob is not None:
            min_logit = _prob_to_logit(float(floor_prob))
            min_tensor = torch.tensor(min_logit, device=device, dtype=logits.dtype)
            logits.data = torch.maximum(logits.data, min_tensor)

        cap_prob = maint_cfg.get('opacity_cap', None)
        if cap_prob is not None:
            max_logit = _prob_to_logit(float(cap_prob))
            max_tensor = torch.tensor(max_logit, device=device, dtype=logits.dtype)
            logits.data = torch.minimum(logits.data, max_tensor)

        if active_mask is None:
            active_mask = torch.zeros_like(last_seen, dtype=torch.bool, device=device)
        else:
            active_mask = active_mask.to(device)

        if active_mask.any():
            boost = float(maint_cfg.get('recency_boost', 0.0))
            if boost != 0.0:
                if logits.dim() == 1:
                    logits.data[active_mask] += boost
                else:
                    logits.data[active_mask] += boost
            updated_last_seen = last_seen.clone()
            updated_last_seen[active_mask] = frame_tensor
            last_seen = updated_last_seen

            updated_hits = visibility_hits.clone()
            updated_hits[active_mask] += 1.0
            visibility_hits = updated_hits

        variables['last_seen'] = last_seen
        variables['visibility_hits'] = visibility_hits
        variables['frame_idx'] = frame_tensor


def _cull_stale_gaussians(params: Dict, variables: Dict, optimizer, frame_idx: int,
                          maint_cfg: Dict, time_idx: int, config: Dict) -> Dict:
    if not maint_cfg.get('enable', False):
        return params, variables
    if 'last_seen' not in variables:
        return params, variables

    device = params['logit_opacities'].device
    num_pts = params['logit_opacities'].shape[0]
    def _resize_tensor(t, target_len, fill_value=0.0):
        if t.shape[0] == target_len:
            return t
        if t.shape[0] > target_len:
            return t[:target_len]
        pad_shape = (target_len - t.shape[0],) + t.shape[1:]
        pad = torch.full(pad_shape, fill_value, device=t.device, dtype=t.dtype)
        return torch.cat((t, pad), dim=0)

    variables['last_seen'] = _resize_tensor(variables['last_seen'], num_pts, fill_value=-1.0)
    variables['visibility_hits'] = _resize_tensor(variables['visibility_hits'], num_pts, fill_value=0.0)

    to_remove = torch.zeros(num_pts, dtype=torch.bool, device=device)
    frame_tensor = torch.tensor(float(frame_idx), device=device, dtype=variables['last_seen'].dtype)

    stale_frames = int(maint_cfg.get('visibility_prune_frames', 0))
    if stale_frames > 0:
        last_seen = variables['last_seen']
        stale_mask = (last_seen >= 0.0) & ((frame_tensor - last_seen) >= float(stale_frames))
        min_hits = maint_cfg.get('visibility_min_hits', 0)
        if min_hits and min_hits > 0:
            stale_mask &= (variables['visibility_hits'] <= float(min_hits))
        to_remove |= stale_mask

    near_thresh = float(maint_cfg.get('near_camera_thresh', 0.0))
    if near_thresh > 0.0:
        with torch.no_grad():
            if config['deforms']['use_deformations']:
                local_means, _, _, _, _ = deform_gaussians(
                    params, time_idx, deform_grad=False, deformation_type=config['deforms']['deform_type']
                )
            else:
                local_means = params['means3D'].detach()
            transformed_pts = transform_to_frame(local_means, params, time_idx,
                                                 gaussians_grad=False, camera_grad=False)
            dist = torch.norm(transformed_pts, dim=1)
            if to_remove.shape[0] != dist.shape[0]:
                to_remove = _resize_tensor(to_remove, dist.shape[0], fill_value=False)
                to_remove |= (dist < near_thresh)

    if to_remove.any():
        print(f"Gaussian maintenance removed {int(to_remove.sum())} elements (stale/near-camera).")
        params, variables = remove_points(to_remove, params, variables, optimizer)

    return params, variables


class GaussianRegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = EncoderBlock(4,32)    # 336x336x4 --> 168x168x32
        self.e2 = EncoderBlock(32,64)   # 168x168x32 --> 84x84x64
        self.e3 = EncoderBlock(64,128)  # 84x84x64 --> 42x42x128
        self.e4 = EncoderBlock(128,256) #  42x42x128 --> 21x21x256


        self.b = Conv_Block(256,512)    # 21x21x256 --> 21x21x512

        self.d1 = DecoderBlock(512,256) # 21x21x512 --> 42x42x256
        self.d2 = DecoderBlock(256,128) # 42x42x256 --> 84x84x128
        self.d3 = DecoderBlock(128,64)  # 84x84x128 --> 168x168x64
        self.d4 = DecoderBlock(64,32)   # 168x168x64 --> 336x336x32

        self.classifier = nn.Conv2d(32,8,kernel_size=1,padding = 0) 
        self.output = nn.Sigmoid()

    def forward(self,inputs):

        s1,p1 = self.e1(inputs)     # 336x336x4 --> 168x168x32
        s2,p2 = self.e2(p1)         # 168x168x32 --> 84x84x64
        s3,p3 = self.e3(p2)         # 84x84x64 --> 42x42x128
        s4,p4 = self.e4(p3)         #  42x42x128 --> 21x21x256

        b = self.b(p4)              # 21x21x256 --> 21x21x512
        
        d1 = self.d1(b,s4)          # 21x21x512 --> 42x42x256
        d2 = self.d2(d1,s3)         # 42x42x256 --> 84x84x128
        d3 = self.d3(d2,s2)         # 84x84x128 --> 168x168x64
        d4 = self.d4(d3,s1)         # 168x168x64 --> 336x336x32
        cls = self.classifier(d4)   # 336x336x32 --> 336x336x8
        return cls


class Conv_Block(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,padding = 1):
        super().__init__()
        self.conv_1 = nn.Conv2d(ch_in,ch_out, kernel_size = kernel_size, padding = padding)
        self.conv_2 = nn.Conv2d(ch_out,ch_out,kernel_size = kernel_size, padding = padding)


        self.batchnorm1 = nn.BatchNorm2d(ch_out)
        self.batchnorm2 = nn.BatchNorm2d(ch_out)

        self.relu = nn.ReLU()

    def forward(self,inputs):
        x = self.conv_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batchnorm2(x)
        return self.relu(x)



class EncoderBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv = Conv_Block(ch_in,ch_out)
        self.pool = nn.MaxPool2d((2,2))


    def forward(self,inputs):
        x = self.conv(inputs)
        p = self.pool(x)


        return x,p
    

class DecoderBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in,ch_out,kernel_size=2,stride = 2,padding = 0) # Strided transpose conv with no padding doubles size
        self.conv = Conv_Block(2*ch_out,ch_out) # We concatenate the skip connection, so we get 2x ch_out channels

    def forward(self, inputs, skip):
        x = self.up(inputs)
        
        # Resize skip to match x
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        
        x = torch.cat([x, skip], axis=1)
        return self.conv(x)

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scared"]:
        return ScaredDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["endonerf"]:
        return EndoNerfDataset(config_dict, basedir, sequence, **kwargs)    
    elif config_dict["dataset_name"].lower() in ['rarp']:
        return RARPDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['hamlyn']:
        return HamlynDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['stereomis']:
        return StereoMisDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def _set_requires_grad_for_phase(params_dict, lr_dict):
    for name, tensor in params_dict.items():
        if not isinstance(name, str):  # safety
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        req = (lr_dict.get(name, 0.0) > 0.0)
        if tensor.requires_grad != req:
            tensor.requires_grad_(req)

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


# def initialize_deformations(params,nr_basis,use_distributed_biases,total_timescale= None):
#     """Function to initialize deformation parameters

#     Args:
#         params (dict): dict containing all the gaussian parameters
#         nr_basis (int): nr of basis functions to generate for each gaussian attribute
#         total_timescale (int): the total timescale of the deformation model, used to evenly distribute biases across timespan

#     Returns:
#         params (dict): Updated params dict with the gaussian deformation parameters
#     """
#     # Means3D, unnorm rotations and log_scales should receive deformation params
#     N = params['means3D'].shape[0]
#     weights = torch.randn([N,nr_basis,10],requires_grad = True,device = 'cuda')*0.0 # We have N x nr_basis x 10 (xyz,scales,rots) weights
#     stds = torch.ones([N,nr_basis,10],requires_grad = True,device = 'cuda')/0.1 # We have N x nr_basis x 10 (xyz,scales,rots) weights
#     if not use_distributed_biases:
#         biases = torch.randn([N,nr_basis,10],requires_grad = True,device = 'cuda')*0.0 # We have N x nr_basis x 10 (xyz,scales,rots) weights
    
    
#     else:
#         interval = torch.ceil(torch.tensor(total_timescale/nr_basis)) # For biases, we want to evenly distribute them across the timespan, so we have nr_basis biases at an interval of total_timescale/nr_basis
#         arange = torch.arange(0,total_timescale,interval).unsqueeze(0).unsqueeze(-1)
#         biases = torch.tile(arange,(N,1,10)) # Repeat the distributed biases to generate basis functions for each parameter

#     params['deform_weights'] = weights
#     params['deform_stds'] = stds
#     params['deform_biases'] = biases


#     return params

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
#     # print(f'xyz: {torch.sum(deform_xyz)}')
#     # print(torch.sum(deform_rots).item())
#     # print(torch.sum(deform_scales).item())
#     xyz = params['means3D']+deform_xyz
#     rots = params['unnorm_rotations']+deform_rots
#     scales = params['log_scales']+deform_scales

#     return xyz,rots,scales

# def deform_gaussians(params, time, deform_grad, N=4):
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
#     if deform_grad:
#         weights = params['deform_weights']
#         stds = params['deform_stds']
#         biases = params['deform_biases']
#     else:
#         weights = params['deform_weights'].detach()
#         stds = params['deform_stds'].detach()
#         biases = params['deform_biases'].detach()

#     # Calculate the absolute difference between time and biases
#     time_diff = torch.abs(time - biases)

#     # Get the indices of the N smallest time differences
#     _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

#     # Create a mask to select only the top N basis functions
#     mask = torch.zeros_like(time_diff, dtype=torch.float)
#     mask.scatter_(1, top_indices, 1.0)

#     # Apply the mask to weights and biases
#     masked_weights = weights * mask
#     masked_biases = biases * mask

#     # Calculate deformations
#     deform = torch.sum(
#         masked_weights * torch.exp(-1 / (2 * stds**2) * (time - masked_biases)**2), dim=1
#     )  # Nx10 gaussians deformations

#     deform_xyz = deform[:, :3]
#     deform_rots = deform[:, 3:7]
#     deform_scales = deform[:, 7:10]

#     xyz = params['means3D'] + deform_xyz
#     rots = params['unnorm_rotations'] + deform_rots
#     scales = params['log_scales'] + deform_scales

#     return xyz, rots, scales


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
#         if True:
#             if deform_grad:
#                 weights = params['deform_weights']
#                 stds = params['deform_stds']
#                 biases = params['deform_biases']
#             else:
#                 weights = params['deform_weights'].detach()
#                 stds = params['deform_stds'].detach()
#                 biases = params['deform_biases'].detach()

#             # Calculate the absolute difference between time and biases
#             time_diff = torch.abs(time - biases)

#             # Get the indices of the N smallest time differences
#             _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

#             # Create a mask to select only the top N basis functions
#             mask = torch.zeros_like(time_diff, dtype=torch.float)
#             mask.scatter_(1, top_indices, 1.0)

#             # Apply the mask to weights and biases
#             masked_weights = weights * mask
#             masked_biases = biases * mask

#             # Calculate deformations
#             deform = torch.sum(
#                 masked_weights * torch.exp(-1 / (2 * stds**2) * (time - masked_biases)**2), dim=1
#             )  # Nx10 gaussians deformations

#             deform_xyz = deform[:, :3]
#             deform_rots = deform[:, 3:7]
#             deform_scales = deform[:, 7:10]
#         else:
#             if deform_grad:
#                 weights = params['deform_weights']
#                 stds = params['deform_stds']
#                 biases = params['deform_biases']
#             else:
#                 weights = params['deform_weights'].detach()
#                 stds = params['deform_stds'].detach()
#                 biases = params['deform_biases'].detach()

#             # Calculate the absolute difference between time and biases
#             time_diff = torch.abs(time - biases)

#             # Get the indices of the N smallest time differences
#             _, top_indices = torch.topk(-time_diff, N, dim=1)  # Negative for smallest values

#             # Create a mask to select only the top N basis functions
#             mask = torch.zeros_like(time_diff, dtype=torch.float)
#             mask.scatter_(1, top_indices, 1.0).detach()

#             # Register a gradient hook to zero out gradients for irrelevant basis functions
#             if deform_grad:
#                 def zero_out_irrelevant_gradients(grad):
#                     return grad * mask

#                 weights.register_hook(zero_out_irrelevant_gradients)
#                 biases.register_hook(zero_out_irrelevant_gradients)
#                 stds.register_hook(zero_out_irrelevant_gradients)

#             # Calculate deformations
#             deform = torch.sum(
#                 weights * torch.exp(-1 / (2 * stds**2) * (time - biases)**2), dim=1
#             )  # Nx10 gaussians deformations

#             deform_xyz = deform[:, :3]
#             deform_rots = deform[:, 3:7]
#             deform_scales = deform[:, 7:10]

#         xyz = params['means3D'] + deform_xyz
#         rots = params['unnorm_rotations'] + deform_rots
#         scales = params['log_scales'] + deform_scales
#         opacities = params['logit_opacities']
#         colors = params['rgb_colors']


#     elif deformation_type == 'simple':
#         # with torch.no_grad():
#         xyz = params['means3D']
#         rots = params['unnorm_rotations']
#         scales = params['log_scales']
#         opacities = params['logit_opacities']
#         colors = params['rgb_colors']

#     return xyz, rots, scales,opacities, colors

def to01(t):
    t = t.permute(2, 0, 1).float()
    # only divide if data is 0..255
    if t.max() > 1.5:
        t = t / 255.0
    return t.clamp_(0, 1)

def initialize_simple_deformations(params, num_frames):
    '''
    This function initializes gaussians that simply update each frame, so we copy the means3D, unnorm rotations and log_scales for each frame and will update over them separately
    '''

    params['means3D'] = params['means3D'][...,None].tile(1,1,num_frames)
    params['unnorm_rotations'] = params['unnorm_rotations'][...,None].tile(1,1,num_frames)
    params['log_scales'] = params['log_scales'][...,None].tile(1,1,num_frames)
    params['logit_opacities'] = params['logit_opacities'][...,None].tile(1,1,num_frames)
    params['rgb_colors'] = params['rgb_colors'][...,None].tile(1,1,num_frames)


    return params

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True,use_deforms = True,deform_type = 'gaussian',nr_basis = 10,use_distributed_biases = False,total_timescale = None,cam = None,random_initialization = False,init_scale = 0.1):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if not random_initialization:
        params = {
            'means3D': means3D,
            'rgb_colors': init_pt_cld[:, 3:6],
            'unnorm_rotations': torch.tensor(unnorm_rots,dtype=torch.float).cuda(),
            'logit_opacities': logit_opacities,
            'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
        }
    else:
        params = {
            'means3D': means3D,
            'rgb_colors': init_pt_cld[:, 3:6],
            'unnorm_rotations': torch.zeros_like(torch.tensor(unnorm_rots),dtype=torch.float).cuda(),
            'logit_opacities': logit_opacities,
            'log_scales': torch.ones_like(torch.tensor(means3D),dtype=torch.float).cuda()*init_scale,
        }

    


                                                                                           
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    if use_deforms:
        if deform_type in ('gaussian', 'deformgs'):
            params = initialize_deformations(
                params,
                nr_basis,
                use_distributed_biases=use_distributed_biases,
                total_timescale=total_timescale
            )
        elif deform_type == 'cv':
            params = initialize_cv_deformations(params)

 
    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))


        
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'last_seen': torch.full((params['means3D'].shape[0],), -1.0, device="cuda").float(),
                 'visibility_hits': torch.zeros(params['means3D'].shape[0], device="cuda").float(),
                 'frame_idx': torch.tensor(0.0, device="cuda").float(),
                 'motion_state': {
                     'prev_tran': None,
                     'pause_counter': 0,
                     'is_contraction': False,
                     'delta': 0.0,
                     'last_step': 0.0,
                     'forward_dir': None
                 }}
    scene_state = {
        "mode": "protect",          # "protect" | "slam"
        "open_frames": 0,
        "recent_big_adds": 0,
    }
    variables["scene_state"] = scene_state
    # If we use simple deformations, we want a list of all parameters for easier optimization
    if use_deforms:
        if deform_type =='simple':
            param_list = [params for _ in range(num_frames)]
        else:
            param_list = params
    else:
        param_list = params
    return param_list, variables


def initialize_optimizer(params, lrs_dict):
    param_groups = []

    for k, v in params.items():
        # Skip SH features if you keep them separate
        #if k == 'feature_rest':
        #    continue

        # Guard: only accept string keys (the LR dict has string keys)
        if not isinstance(k, str):
            #print(f"[initialize_optimizer] Skipping non-string param key: {k!r}")
            continue

        lr = lrs_dict.get(k, None)
        if lr is None:
            # No LR provided: skip (or set lr = 0.0 to freeze instead)
            print(f"[initialize_optimizer] No LR for '{k}', skipping this param.")
            continue

        param_groups.append({'params': [v], 'name': k, 'lr': float(lr)})
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


# def grn_initialization(model,params,init_pt_cld,mean3_sq_dist,color,depth,mask = None,cam= None):

#     normalize = torchvision.transforms.Normalize([0.46888983, 0.29536288, 0.28712815],[0.24689102 ,0.21034359, 0.21188641])
#     inv_normalize = torchvision.transforms.Normalize([-0.46888983/0.24689102,-0.29536288/0.21034359,-0.28712815/0.21188641],[1/0.24689102,1/0.21034359,1/0.21188641]) #Take the inverse of the normalization

#     color = normalize(color).detach()
#     scale = torch.max(depth).detach()
#     depth = (depth/scale).detach()

#     input = torch.cat((color,depth),axis = 0).unsqueeze(0).detach()
#     if mask == None:
#         mask = depth > 0
#         mask = mask.tile(1,3,1,1)
#         mask = mask[0,0,:,:].reshape(-1)
        

#     output = model(input).detach()
#     # print(output)
#     cols = torch.permute(output[0], (1, 2, 0)).reshape(-1, 8) # (C, H, W) -> (H, W, C) -> (H * W, C)
#     rots = cols[:,:4]

#     scales_norm = cols[:,4:7]
#     opacities = cols[:,7][:,None]

#     # local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,0,False,5,'simple')
#     # rendervar = transformed_GRNparams2rendervar(params,local_means,local_rots,local_scales,local_opacities,local_colors)   

#     # im,radius,_ = Renderer(raster_settings=cam)(**rendervar)
#     # plt.imshow(im.permute(1,2,0).cpu().detach())  
#     # plt.title('Before grn_init')         
#     # plt.show() 
#     # If we use simple deformations, rotations and scales will have shape [C x Num_gaussians x num_frames],
#     # We need to apply the GRN inialization to each timestep
#     if len(params['unnorm_rotations'].shape) ==3:
#         params['unnorm_rotations'] = (rots[mask])[...,None].tile(1,1,params['unnorm_rotations'].shape[2])
#         params['log_scales'] = (scales_norm[mask]*mean3_sq_dist[:,None].tile(1,3))[...,None].tile(1,1,params['log_scales'].shape[2])
#         params['logit_opacities'] = (opacities[mask])[...,None].tile(1,1,params['logit_opacities'].shape[2])
#     else:
#         params['unnorm_rotations'] = rots[mask]
#         params['log_scales'] = scales_norm[mask]*mean3_sq_dist[:,None].tile(1,3)
#         params['logit_opacities'] = opacities[mask]
    

#     # local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,0,False,5,'simple')
#     # rendervar = transformed_GRNparams2rendervar(params,local_means,local_rots,local_scales,local_opacities,local_colors)    

#     # im,radius,_ = Renderer(raster_settings=cam)(**rendervar)
#     # plt.imshow(im.permute(1,2,0).cpu().detach())  
#     # plt.title('After grn_init')         
#     # plt.show() 
#     return params

def _clamp_cv_deform(params, config):
    # Only if you enabled CV-style fields in slam_helpers.initialize_cv_deformations(...)
    for k in ('cv_vel_xyz','cv_vel_log_scales','cv_angvel_aa'):
        if k not in params: 
            continue
    with torch.no_grad():
        if 'cv_vel_xyz' in params:
            mv = float(config['deforms'].get('max_vel_xyz', 0.05))
            params['cv_vel_xyz'].clamp_(-mv, mv)
        if 'cv_vel_log_scales' in params:
            ms = float(config['deforms'].get('max_logscale_vel', 0.02))
            params['cv_vel_log_scales'].clamp_(-ms, ms)
        if 'cv_angvel_aa' in params:
            ma = float(config['deforms'].get('max_ang_vel', 0.2))
            w = params['cv_angvel_aa']
            n = torch.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(ma / n, max=1.0)
            params['cv_angvel_aa'].mul_(scale)

  
# replace BOTH of these occurrences in main_SurgeSplat:
#   color = color.permute(2, 0, 1) / 255
#   tracking_color = tracking_color.permute(2, 0, 1) / 255

def to01(t):
    t = t.permute(2, 0, 1).float()
    # Only divide if it’s actually 0..255
    if t.max() > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)




def initialize_first_timestep(color,depth,intrinsics,pose, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, use_simplification=True,use_gt_depth = True,
                              use_deforms=True,deform_type='gaussian',nr_basis = 10,use_distributed_biases = False,total_timescale=None,use_grn=False,grn_model=None,
                              random_initialization=False,init_scale=0.1,reduce_gaussians= True,reduction_type = 'laplace',reduction_fraction = 0.8):
    # Get RGB-D Data & Camera Parameters
    # color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    # color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    # depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.eye(4, device=pose.device, dtype=pose.dtype)
    #w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = to01(color)                                  # fixes curr_data['im'] path
        
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    print("Initial gaussian mask contains {} valid pixels".format(torch.sum(mask)))
    mask = mask.reshape(-1)
    if reduce_gaussians:
        mask = get_mask(mask,color,reduction_type,reduction_fraction)
    print("After reducing gaussians, mask conttains {} valid pixels".format(torch.sum(mask)))



    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters

  
    params_list, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification,use_deforms=use_deforms,deform_type=deform_type,nr_basis = nr_basis,use_distributed_biases=use_distributed_biases,total_timescale=total_timescale,cam = cam,random_initialization=random_initialization,init_scale=init_scale)
    
    # bloat_params = True
    # if bloat_params:
    #     if isinstance(params_list,list):
    #         params = params_list[0]
    #         params['bloated_params'] = torch.zeros(init_pt_cld.shape[0],1,device='cuda')
    #         params_list = [params for _ in range(num_frames)]
    #     else:
    #         params['bloated_params'] = torch.zeros(init_pt_cld.shape[0],1,device='cuda')
    #         params_list = params
    
    
    
    if use_grn:
        if isinstance(params_list,list):
            params = grn_initialization(grn_model,params_list[0],init_pt_cld,mean3_sq_dist,color,depth,mask,cam = cam)
            params_list = [params for _ in range(num_frames)]
        else:
            params = grn_initialization(grn_model,params_list,init_pt_cld,mean3_sq_dist,color,depth,mask,cam = cam)
            params_list = params


    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio # NOTE: change_here

    if densify_dataset is not None:
        return params_list, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params_list, variables, intrinsics, w2c, cam
    


def get_loss(params, params_initial, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None,use_gt_depth = True,gaussian_deformations = True,save_idx=0,
             use_grn = False,deformation_type = 'gaussian', gate_thresh = None):

    global w2cs, w2ci
    # Initialize Loss Dictionary
    losses = {}
    gate_thresh = 0.0 if gate_thresh is None else gate_thresh
    if gaussian_deformations: # If we train for deformations, the location of the means depends on the timestep
        if tracking:
            local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,iter_time_idx,deform_grad = True,deformation_type = deformation_type)
            # raise ValueError('This shouldnt really happen tbh')
            # print(torch.sum(local_means-params['means3D']))
        else:
            local_means,local_rots,local_scales,local_opacities,local_colors= deform_gaussians(params,iter_time_idx,deform_grad = False,deformation_type = deformation_type)
    else:
        local_means = params['means3D']
        local_rots = params['unnorm_rotations']
        local_scales = params['log_scales']
        local_opacities = params['logit_opacities']
        local_colors = params['rgb_colors']
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        
        transformed_pts = transform_to_frame(local_means,params, iter_time_idx, 
                                             gaussians_grad=True,
                                             camera_grad=True)
    elif mapping:
        if do_ba: # Bundle Adjustment
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(local_means,params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(local_means,params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(local_means,params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)
    

    
    w_compact = float(loss_weights.get('xyzt_compact', 1e-4))   # small!
    w_center  = float(loss_weights.get('xyzt_center',  1e-4))
    if 't_mu' in params:
        if w_compact > 0.0 or w_center > 0.0:
            mu = params['t_mu']
            logvar = params['t_logvar']
            var = torch.exp(logvar)

            # Encourage small temporal support (but don't crush to zero)
            compact = (var.sqrt()).mean()   # mean sigma_t

            # Encourage center near current frame if it contributes (softly)
            # Use gate as responsibility (stopgrad on gate to avoid trivial tricks)
            with torch.no_grad():
                gate = xyzt_time_gate(params, float(iter_time_idx))
                gate = gate / (gate.sum() + 1e-8)
            center = (gate * (mu - float(iter_time_idx)).abs()).sum()

            losses['xyzt_compact'] = compact
            losses['xyzt_center']  = center


    bloat_params = False
    # print("We got to the loss calculation")
    if bloat_params: # For ablation study: add bloating parameters to computation graph to see effect on optimization time
        
        # bloat_test = torch.zeros_like(local_opacities,device='cuda')
        bloat = torch.zeros(local_opacities.shape[0],1200,device='cuda')
        bloat = torch.sum(bloat)
        # print(bloat.shape)
        local_opacities = local_opacities+bloat
    # Initialize Render Variables
    # print()
    time_gate = None
    if 't_mu' in params:    
        time_gate = xyzt_time_gate(params, float(iter_time_idx))     # time index of this frame

    # ---- predicted-depth-aware cull (training) ----
    scene_state = variables.get("scene_state", {"mode": "slam"})
    if scene_state.get("mode", "slam") == "slam":
        pred_depth = curr_data['depth']
        if not torch.is_tensor(pred_depth):
            pred_depth = torch.as_tensor(pred_depth, device=transformed_pts.device, dtype=transformed_pts.dtype)
        else:
            pred_depth = pred_depth.to(device=transformed_pts.device, dtype=transformed_pts.dtype)
        pred_depth = pred_depth.detach()
        if pred_depth.dim() == 2:
            pred_depth = pred_depth.unsqueeze(0)
        elif pred_depth.dim() == 3 and pred_depth.shape[0] > 1:
            pred_depth = pred_depth[:1]

        intrinsics_tensor = curr_data['intrinsics']
        if torch.is_tensor(intrinsics_tensor):
            intrinsics_tensor = intrinsics_tensor.to(device=transformed_pts.device, dtype=transformed_pts.dtype)
        else:
            intrinsics_tensor = torch.as_tensor(intrinsics_tensor, device=transformed_pts.device, dtype=transformed_pts.dtype)

        fx = intrinsics_tensor[0, 0]
        fy = intrinsics_tensor[1, 1]
        cx = intrinsics_tensor[0, 2]
        cy = intrinsics_tensor[1, 2]

        pts_cam = transformed_pts
        z_vals = pts_cam[:, 2].clamp_min(1e-4)
        u = (pts_cam[:, 0] * fx / z_vals) + cx
        v = (pts_cam[:, 1] * fy / z_vals) + cy

        H, W = pred_depth.shape[1], pred_depth.shape[2]
        in_w = (u >= 0) & (u <= (W - 1))
        in_h = (v >= 0) & (v <= (H - 1))
        pix_valid = in_w & in_h

        if pix_valid.any():
            u_pix = torch.clamp(u[pix_valid], 0, W - 1).long()
            v_pix = torch.clamp(v[pix_valid], 0, H - 1).long()
            pred_z = pred_depth[0, v_pix, u_pix]
            tol = 0.03
            valid_depth_mask = pred_z > 1e-4
            depth_keep = torch.ones_like(pred_z, dtype=torch.bool)
            depth_keep[valid_depth_mask] = z_vals[pix_valid][valid_depth_mask] <= (pred_z[valid_depth_mask] + tol)
            final_keep = torch.ones_like(pix_valid, dtype=torch.bool)
            final_keep[pix_valid] = depth_keep
            num_valid = int(pix_valid.sum().item())
            if num_valid > 0:
                min_keep = max(1, int(0.1 * num_valid))
                if final_keep.sum().item() < min_keep:
                    tmp = final_keep.clone()
                    tmp[pix_valid] = True
                    final_keep = tmp
            if final_keep.sum().item() == 0:
                final_keep = pix_valid

            transformed_pts = transformed_pts[final_keep]
            local_means     = local_means[final_keep]
            local_rots      = local_rots[final_keep]
            local_scales    = local_scales[final_keep]
            local_opacities = local_opacities[final_keep]
            local_colors    = local_colors[final_keep]
            if time_gate is not None:
                time_gate = time_gate[final_keep]

    # --- camera-space handling of near/front gaussians ---
    # transformed_pts is in *current camera* frame because you called transform_to_frame(...)
    # we fade gaussians that drift close to the lens, and still hard-zero those that
    # are both too close and not strongly gated for this frame.
    cam_z = transformed_pts[:, 2]

    # Soft fade out gaussians that get very close to the camera regardless of gating.
    fade_near, fade_far = 0.02, 0.04
    local_opacities_modified = False

    fade_zone = (cam_z < fade_far) & (cam_z >= fade_near)
    if fade_zone.any():
        local_opacities = local_opacities.clone()
        local_opacities_modified = True
        fade = (cam_z[fade_zone] - fade_near) / (fade_far - fade_near)
        fade = torch.log(fade + 1e-6) * 2.0
        if fade.dim() == 1 and local_opacities[fade_zone].dim() > 1:
            fade = fade.unsqueeze(-1)
        local_opacities[fade_zone] += fade

    too_close_all = cam_z < fade_near
    if too_close_all.any():
        if not local_opacities_modified:
            local_opacities = local_opacities.clone()
            local_opacities_modified = True
        local_opacities[too_close_all] = -10.0

    near_thresh = 0.025  # ~2.5 cm in your colon scale, tune 0.015-0.04

    # if we have time_gate, use it, otherwise pretend everything is active
    if time_gate is not None:
        active = (time_gate > gate_thresh).reshape(-1)
    else:
        active = torch.ones_like(cam_z, dtype=torch.bool)

    too_close = cam_z < near_thresh
    kill_mask = too_close & (~active)

    if kill_mask.any():
        # hard-zero their opacities for this frame
        if not local_opacities_modified:
            local_opacities = local_opacities.clone()
            local_opacities_modified = True
        local_opacities[kill_mask] = -10.0  # sigmoid(-10) ~ 0

    if use_grn:
        rendervar = transformed_GRNparams2rendervar(params, transformed_pts,local_rots,local_scales,local_opacities,local_colors)
        depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_pts,local_rots,local_scales,local_opacities)
    else:
        rendervar = transformed_params2rendervar(params, transformed_pts,local_rots,local_scales,local_opacities,local_colors)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_pts,local_rots,local_scales,local_opacities)
    
    # Apply temporal gating if available
    if time_gate is not None:
        rendervar = apply_xyzt_gate(rendervar, time_gate, gate_thresh=gate_thresh)
        depth_sil_rendervar = apply_xyzt_gate(depth_sil_rendervar, time_gate, gate_thresh=gate_thresh)
    
    # Inter-frame consistency loss
    prev_means = variables.get('prev_local_means', None)
    prev_gate = variables.get('prev_time_gate', None)
    if iter_time_idx > 0 and prev_means is not None:
        min_gauss = min(prev_means.shape[0], local_means.shape[0])
        if min_gauss > 0:
            curr_means = local_means[:min_gauss]
            prev_means_trim = prev_means[:min_gauss]
            curr_gate = time_gate[:min_gauss] if time_gate is not None else torch.ones(min_gauss, device=curr_means.device, dtype=curr_means.dtype)
            prev_gate_trim = prev_gate[:min_gauss] if (prev_gate is not None) else torch.ones_like(curr_gate)
            gate_weights = curr_gate * prev_gate_trim
            losses['inter_frame'] = ((curr_means - prev_means_trim) ** 2 * gate_weights.view(-1, 1)).sum(-1).mean()

    # RGB Rendering
    try:
        rendervar['means2D'].retain_grad()
    except:
        pass
    
 

    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    inter_w = float(loss_weights.get('inter_frame_rgb', 0.0))
    if inter_w > 0.0 and iter_time_idx > 0:
        with torch.no_grad():
            prev_local_means_full, prev_rots_full, prev_scales_full, prev_opac_full, prev_cols_full = \
                deform_gaussians(params, iter_time_idx - 1, False, deformation_type=deformation_type)

            prev_transformed_pts = transform_to_frame(
                prev_local_means_full, params, iter_time_idx,
                gaussians_grad=False, camera_grad=False
            )

            prev_rendervar = transformed_params2rendervar(
                params,
                prev_transformed_pts,
                prev_rots_full,
                prev_scales_full,
                prev_opac_full,
                prev_cols_full
            )
            if 't_mu' in params:
                prev_gate_full = xyzt_time_gate(params, float(iter_time_idx - 1))
                prev_rendervar = apply_xyzt_gate(prev_rendervar, prev_gate_full, gate_thresh=gate_thresh)

            prev_im, _, _ = Renderer(raster_settings=curr_data['cam'])(**prev_rendervar)

        rgb_cons = (im - prev_im).abs().mean()
        losses['inter_frame_rgb'] = rgb_cons

    variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification
    # plt.imshow(im.permute(1,2,0).cpu().detach())
    # plt.show()
    
    # Depth & Silhouette Rendering
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    bg_mask = energy_mask(curr_data['im'])
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 20*depth_error.mean())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask & bg_mask
    
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask
    if not use_gt_depth:
        rendered_depth_aligned,predicted_depth_aligned,_,_,_,_ = align_shift_and_scale(depth,curr_data['depth'],mask)
        # rendered_depth_aligned = (depth-depth.min())/(depth.max()-depth.min())+0.1
        # predicted_depth_aligned = (curr_data['depth']-curr_data['depth'].min())/(curr_data['depth'].max()-curr_data['depth'].min())+0.1
        # rendered_depth_aligned = depth
        # predicted_depth_aligned = curr_data['depth']
    
    if False:
        fig,ax = plt.subplots(2,4)
        ax[0,0].imshow(im.permute(1,2,0).cpu().detach())
        ax[0,0].set_title('Rendered im')
        ax[0,1].imshow(curr_data['im'].permute(1,2,0).cpu().detach())
        ax[0,1].set_title('Input img')
        im0 = ax[1,0].imshow(rendered_depth_aligned.squeeze().cpu().detach())
        plt.colorbar(im0,ax = ax[1,0])
        ax[1,0].set_title('Rendered depth')
        ax[1,1].imshow(predicted_depth_aligned.squeeze().cpu().detach())
        ax[1,1].set_title('Input depth')
        ax[0,2].imshow(nan_mask.squeeze().cpu().detach())
        ax[0,2].set_title('Nan mask')
        ax[0,3].imshow(bg_mask.squeeze().cpu().detach())
        ax[0,3].set_title('BG mask')
        ax[1,2].imshow(mask.squeeze().cpu().detach())
        ax[1,2].set_title('Mask')
        im1 = ax[1,3].imshow(rendered_depth_aligned.squeeze().cpu().detach()-predicted_depth_aligned.squeeze().cpu().detach())
        ax[1,3].set_title('Depth diff aligned')
        plt.colorbar(im1,ax = ax[1,3])
        plt.show()
    if not save_idx == None:
        ii = curr_data['id']
        img = Image.fromarray((im.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
        os.makedirs(f'./scripts/plots/{ii}',exist_ok=True)
        img.save(f'./scripts/plots/{ii}/{save_idx}.png')



    if not save_idx == None:
        ii = curr_data['id']
        diff = torch.abs(im-curr_data['im'])


        # c_d_norm = (curr_data['depth']-curr_data['depth'].min())/(curr_data['depth'].max()-curr_data['depth'].min())
        # d_norm = (depth-depth.min())/(depth.max()-depth.min())
        # diff_depth = torch.abs(predicted_depth_aligned.squeeze()-rendered_depth_aligned.squeeze())
        img = torch.cat([im.permute(1,2,0),curr_data['im'].permute(1,2,0),diff.permute(1,2,0)],dim = 1).cpu().detach().numpy()
        img = Image.fromarray((img*255).astype(np.uint8))



        # rendered_depth_norm = (rendered_depth_aligned-rendered_depth_aligned.min())/(rendered_depth_aligned.max()-rendered_depth_aligned.min())
        # predicted_depth_norm = (predicted_depth_aligned-predicted_depth_aligned.min())/(predicted_depth_aligned.max()-predicted_depth_aligned.min())
        # diff_depth_norm = (diff_depth-diff_depth.min())/(diff_depth.max()-diff_depth.min())
        # depth_img = torch.cat([rendered_depth_norm.squeeze(),predicted_depth_norm.squeeze(),diff_depth_norm.squeeze()],dim = 1).cpu().detach().numpy()
        #
        # depth_img = Image.fromarray((depth_img*255).astype(np.uint8))
        
        if tracking:
            os.makedirs(f'./scripts/plots/tracking/rgb/{ii}',exist_ok=True)
            img.save(f'./scripts/plots/tracking/rgb/{ii}/{save_idx}.png')
            os.makedirs(f'./scripts/plots/tracking/depth/{ii}',exist_ok=True)
            # depth_img.save(f'./scripts/plots/tracking/depth/{ii}/{save_idx}.png')
        elif mapping:
            os.makedirs(f'./scripts/plots/mapping/{ii}',exist_ok=True)
            img.save(f'./scripts/plots/mapping/{ii}/{save_idx}.png')
            os.makedirs(f'./scripts/plots/mapping/depth/{ii}',exist_ok=True)
            # depth_img.save(f'./scripts/plots/mapping/depth/{ii}/{save_idx}.png')
    

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking and use_gt_depth:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        elif tracking and not use_gt_depth:
            losses['depth']= torch.abs(rendered_depth_aligned - predicted_depth_aligned)[mask].mean()
            # print(f'using aligned losses, rendered range [{rendered_depth_aligned.min()}-{rendered_depth_aligned.max()}], predicted [{predicted_depth_aligned.min()}-{predicted_depth_aligned.max()}]')
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    
    # Deformation regularization
    if tracking and gaussian_deformations and deformation_type == 'gaussian':
        losses['deform'] = torch.sum(torch.square(params['means3D']-local_means))/params['means3D'].shape[0]
    elif tracking and gaussian_deformations and deformation_type == 'simple': 
        nr_initial_gauss = params_initial['means3D'].shape[0]
        nr_current_gauss = local_means.shape[0]
        nr_gauss = min(nr_initial_gauss,nr_current_gauss)
        losses['deform'] = torch.sum(torch.square(params_initial['means3D'][:nr_gauss]-local_means[:nr_gauss]))/nr_gauss

   


    weighted_losses = {k: v * float(loss_weights.get(k, 0.0)) for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    
    seen = radius > 0

    n = radius.shape[0]  # current number of gaussians rendered (matches params size)

    def _pad_or_trim_1d(t, n, fill=0.0):
        if t.shape[0] == n:
            return t
        if t.shape[0] < n:
            pad = torch.full((n - t.shape[0],), fill, device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)
        else:
            return t[:n]

    for key, fill in [
        ('means2D_gradient_accum', 0.0),
        ('denom', 0.0),
        ('max_2D_radius', 0.0),
        # If you keep a per-point timestep buffer:
        ('timestep', 0.0),
    ]:
        if key in variables:
            variables[key] = _pad_or_trim_1d(variables[key], n, fill)

    # Also ensure mask aligns to current N
    if seen.shape[0] != n:
        seen = seen[:n]





    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    # Cache current state for next iteration's temporal losses
    with torch.no_grad():
        variables['prev_local_means'] = local_means.detach().clone()
        variables['prev_time_gate'] = time_gate.detach().clone() if time_gate is not None else None
        variables['prev_time_idx'] = int(iter_time_idx)
    weighted_losses['loss'] = loss
    # print(weighted_losses)
    return loss, variables, weighted_losses


# def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification,nr_basis = 10,use_distributed_biases = False, total_timescale = None,use_deform = True,deform_type = 'gaussian',num_frames = 1,
#                             random_initialization = False,init_scale = 0.1):
#     num_pts = new_pt_cld.shape[0]
#     means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
#     unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
#     logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
#     if not random_initialization:
#         params = {
#             'means3D': means3D,
#             'rgb_colors': new_pt_cld[:, 3:6],
#             'unnorm_rotations': torch.tensor(unnorm_rots,dtype=torch.float).cuda(),
#             'logit_opacities': logit_opacities,
#             'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
#         }
#     else:
#         params = {
#             'means3D': means3D,
#             'rgb_colors': new_pt_cld[:, 3:6],
#             'unnorm_rotations': torch.zeros_like(torch.tensor(unnorm_rots),dtype=torch.float).cuda(),
#             'logit_opacities': logit_opacities,
#             'log_scales': torch.ones_like(torch.tensor(means3D),dtype=torch.float).cuda()*init_scale,
#         }
#     # print(f'num pts {num_pts}')
#     if use_deform and deform_type == 'gaussian':
#         params = initialize_deformations(params,nr_basis = nr_basis,use_distributed_biases=use_distributed_biases,total_timescale = total_timescale)
#     # elif use_deform and deform_type == 'simple':
#     #     params = initialize_simple_deformations(params,num_frames)
#     if not use_simplification:
#         params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
#     for k, v in params.items():
#         # Check if value is already a torch tensor
#         if not isinstance(v, torch.Tensor):
#             params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
#         else:
#             params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

#     return params


# def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True,
#                       nr_basis = 10,use_distributed_biases = False,total_timescale = None,use_grn=False,grn_model=None,
#                       use_deform = True,deformation_type = 'gaussian',num_frames = 1,
#                       random_initialization=False,init_scale=0.1,cam = None):
#     # Silhouette Rendering
#     if use_deform == True:
#         local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,time_idx,True,deformation_type =deformation_type)
#     else:
#         local_means = params['means3D']
#         local_rots = params['unnorm_rotations']
#         local_scales = params['log_scales']
#         local_opacities = params['logit_opacities']
#         local_colors = params['rgb_colors']
    
#     transformed_pts = transform_to_frame(local_means,params, time_idx, gaussians_grad=False, camera_grad=False)
#     if not use_grn:
#         depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
#                                                                     transformed_pts,local_rots,local_scales,local_opacities)
#     else:
#         depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, curr_data['w2c'],
#                                                                     transformed_pts,local_rots,local_scales,local_opacities)
#     depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
#     silhouette = depth_sil[1, :, :]
#     non_presence_sil_mask = (silhouette < sil_thres)
#     # Check for new foreground objects by using GT depth
#     gt_depth = curr_data['depth'][0, :, :]
#     gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())*10+0.01
    


#     render_depth = depth_sil[0, :, :]
#     depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
#     non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
#     # Determine non-presence mask
#     non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
#     # Flatten mask
#     non_presence_mask = non_presence_mask.reshape(-1)

#     # Get the new frame Gaussians based on the Silhouette
#     if torch.sum(non_presence_mask) > 0:


#         # depth_diff = torch.abs(gt_depth - depth_sil[0, :, :])
#         # fig,ax = plt.subplots(1,3)
#         # im0 = ax[0].imshow(gt_depth.squeeze().cpu().detach())
#         # ax[0].set_title('GT depth')
#         # im1 = ax[1].imshow(depth_sil[0].squeeze().cpu().detach())
#         # ax[1].set_title('Rendered depth')
#         # im2 = ax[2].imshow(depth_diff.squeeze().cpu().detach())
#         # ax[2].set_title('Depth diff')
#         # plt.colorbar(im0,ax = ax[0])
#         # plt.colorbar(im1,ax = ax[1])
#         # plt.colorbar(im2,ax = ax[2])
#         # plt.show()
#         # Get the new pointcloud in the world frame
#         curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
#         curr_cam_tran = params['cam_trans'][..., time_idx].detach()
#         curr_w2c = torch.eye(4).cuda().float()
#         curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
#         curr_w2c[:3, 3] = curr_cam_tran
#         valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
#         non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
#         valid_color_mask = energy_mask(curr_data['im']).squeeze()
#         non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
#         new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
#                                     curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
#                                     mean_sq_dist_method=mean_sq_dist_method)
#         new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification,nr_basis = nr_basis,use_distributed_biases=use_distributed_biases,total_timescale = total_timescale,use_deform = use_deform,deform_type=deformation_type,
#                                             num_frames = num_frames,random_initialization=random_initialization,init_scale=init_scale)
#         if use_grn:
#             new_params = grn_initialization(grn_model,new_params,new_pt_cld,mean3_sq_dist,curr_data['im'],curr_data['depth'],non_presence_mask,cam = cam)

#         # # Adding new params happens to all timesteps due to construction of tensors, but they only need to be added to current and future timesteps,
#         # # Therefore means,scales and rotations are set to 0 for previous timesteps
#         # mask = torch.ones((1,1,new_params['means3D'].shape[-1]),device="cuda")
#         # mask[0,0,time_idx-1:] = 0
#         params_iter = params
#         for k, v in new_params.items():
#             # if k == 'logit_opacities':
#             #     params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
#             # else:
#             params_iter[k] = torch.nn.Parameter(torch.cat((params_iter[k], v), dim=0).requires_grad_(True))
#         num_pts = params_iter['means3D'].shape[0]
#         variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
#         variables['denom'] = torch.zeros(num_pts, device="cuda").float()
#         variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
#         new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
#         variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
#     return params_iter, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop, motion_state: Optional[Dict] = None, motion_cfg: Optional[Dict] = None):
    with torch.no_grad():
        if curr_time_idx == 0:
            return params

        prev_rot = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
        prev_tran = params['cam_trans'][..., curr_time_idx-1].detach()

        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(prev_rot)
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = prev_tran
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = prev_rot
            params['cam_trans'][..., curr_time_idx] = prev_tran

        if motion_cfg and motion_cfg.get('enable', False) and motion_state:
            if not motion_state.get('is_contraction', False):
                forward_dir = motion_state.get('forward_dir')
                last_step = motion_state.get('last_step', 0.0)
                if forward_dir is not None and last_step > 0.0:
                    step_gain = motion_cfg.get('forward_step_gain', 1.0)
                    step = last_step * step_gain
                    max_step = motion_cfg.get('max_forward_step')
                    if max_step is not None:
                        step = min(step, max_step)
                    forward_dir_tensor = forward_dir.to(params['cam_trans'].device).view_as(prev_tran)
                    new_tran = prev_tran + forward_dir_tensor * step
                    params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # decay the expected step while paused to avoid stale forward predictions
                motion_state['last_step'] = motion_state.get('last_step', 0.0) * motion_cfg.get('pause_forward_decay', 0.5)
    
    return params

def optimize_initialization(params,params_init,curr_data,num_iters_initialization,variables,iter_time_idx,config, gate_override: Optional[float] = None):
    optimizer = initialize_optimizer(params,config['GRN']['random_initialization_lrs'])
    iter = 0
    save_idx = 0

    progress_bar = tqdm(range(num_iters_initialization), desc=f"Initial optimization")
    active_gate_thresh = config['deforms'].get('xyzt_gate_thresh', 0.0) if gate_override is None else gate_override

    while True:
        loss, variables, losses = get_loss(params, params_init,curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                    config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                    config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True,
                    visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                    tracking_iteration=iter,use_gt_depth = config['depth']['use_gt_depth'],save_idx=None,gaussian_deformations=config['deforms']['use_deformations'],
                    use_grn = config['GRN']['use_grn'],deformation_type = config['deforms']['deform_type'], gate_thresh = active_gate_thresh)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        save_idx +=1

        if iter == num_iters_initialization:
            break
        else:   
            # progress_bar = tqdm(range(num_iters_initialization), desc=f"Tracking Time Step: {time_idx}")
            progress_bar.update(1)
            iter+=1
    progress_bar.close()
    return params 

def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # timer = Timer()
    # timer.start()
    
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if 'distance_keyframe_selection' not in config:
        config['distance_keyframe_selection'] = False
    if config['distance_keyframe_selection']:
        print("Using CDF Keyframe Selection. Note that \'mapping window size\' is useless.")
        if 'distance_current_frame_prob' not in config:
            config['distance_current_frame_prob'] = 0.5
    if 'gaussian_simplification' not in config:
        config['gaussian_simplification'] = True # simplified in paper
    if not config['gaussian_simplification']:
        print("Using Full Gaussian Representation, which may cause unstable optimization if not fully optimized.")
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "train_or_test" not in dataset_config:
        dataset_config["train_or_test"] = 'all'
    if "preload" not in dataset_config:
        dataset_config["preload"] = False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False


    if not config['depth']['use_gt_depth']:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        model = SurgeDepth(**model_configs[config['depth']['model_size']]).cuda()
        model.load_state_dict(torch.load(config['depth']['model_path']))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        # model = SurgeDepth()
    # Poses are relative to the first frame

    if config['GRN']['use_grn']:
        print('Using GRN for initialization')
        

        grn_model = GaussianRegressionNetwork().cuda()
        grn_model.load_state_dict(torch.load(config['GRN']['model_path']))
        grn_model.eval()
        grn_model.requires_grad = False
    else:
        grn_model = None



    print(gradslam_data_cfg)
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        train_or_test=dataset_config["train_or_test"]
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = dataset.end

    if dataset_config["train_or_test"] == 'train': # kind of ill implementation here. train_or_test should be 'all' or 'train'. If 'test', you view test set as full dataset.
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"], # if you eval, you should keep reso as raw image.
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test='test'
        )
    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset, 
                                                                        use_simplification=config['gaussian_simplification'],
                                                                        nr_basis=config['deforms']['nr_basis'],
                                                                        use_distributed_biases=config['deforms']['use_distributed_biases'],
                                                                        total_timescale=config['deforms']['total_timescale'],
                                                                        use_deforms=config['deforms']['use_deformations'],
                                                                        deform_type=config['deforms']['deform_type'],
                                                                        use_grn = config['GRN']['use_grn'])    


    else:
        color, depth, intrinsics, pose = dataset[0]
        # Initialize Parameters & Canoncial Camera parameters
        # plt.imshow(depth.squeeze().cpu().detach())
        # plt.title('gt depth')
        # plt.colorbar()
        # plt.show()
        if not config['depth']['use_gt_depth']:
            color_input = color.permute(2,0,1).unsqueeze(0).cuda()/255 # Change from WxHxC to BxCxWxH for inference
            color_input = torchvision.transforms.functional.normalize(color_input,config['depth']['normalization_means'],config['depth']['normalization_stds']) # Applying normalization
            t_pred = config['depth']['shift_pred']
            s_pred = config['depth']['scale_pred']
            t_gt =   config['depth']['shift_gt']
            s_gt =   config['depth']['scale_gt']
            output = model(color_input)
            output_norm = (output-t_pred) * (s_gt / s_pred) + t_gt
            # t_pred = torch.median(output)
            # s_pred = torch.mean(torch.abs(output-t_pred))
            # output_norm = (output-output.min())/(output.max()-output.min())
            # pred_disp = (output_norm)*s_gt + t_gt +1 # TODO fix this scaling offset
            # # plt.imshow(output.squeeze().cpu().detach())
            # # plt.title('predicted depth')
            # # plt.colorbar()
            # # plt.show()
            # print(pred_disp.min())
            # depth = 1/pred_disp # Convert disp to depth
            # depth = depth.permute(1,2,0) # CxWxH --> WxHxC to align with rest of the pipeline    
            # print(depth.min())
            # output_norm = (output-output.min())/(output.max()-output.min())
            # pred_disp = (output_norm)*s_gt + t_gt +1 # TODO fix this scaling offse0t
            # plt.imshow(output.squeeze().cpu().detach())
            # plt.title('predicted depth')
            # plt.colorbar()
            # plt.show()
            # print(pred_disp.min())
            # depth = 1/pred_disp # Convert disp to depth
            # depth = depth.permute(1,2,0)*10 # CxWxH --> WxHxC to align with rest of the pipeline    
            # print(depth.min())
            depth = 1/(output_norm)
            depth = depth.permute(1,2,0)*10

        # plt.imshow(depth.squeeze().cpu().detach())
        # plt.title('predicted depth')
        # plt.colorbar()
        # plt.show()
        color = to01(color)
        depth = depth.permute(2, 0, 1)

        print(config['deforms']['deform_type'])
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(color, depth, intrinsics, pose, 
                                                                            num_frames, 
                                                                            config['scene_radius_depth_ratio'],
                                                                            config['mean_sq_dist_method'], 
                                                                            use_simplification=config['gaussian_simplification'],
                                                                            use_gt_depth = config['depth']['use_gt_depth'],
                                                                            nr_basis=config['deforms']['nr_basis'],
                                                                            use_distributed_biases=config['deforms']['use_distributed_biases'],
                                                                            total_timescale=config['deforms']['total_timescale'],
                                                                            use_grn=config['GRN']['use_grn'],
                                                                            grn_model=grn_model,
                                                                            use_deforms=config['deforms']['use_deformations'],
                                                                            deform_type=config['deforms']['deform_type'],
                                                                            random_initialization=config['GRN']['random_initialization'],
                                                                            init_scale=config['GRN']['init_scale'],
                                                                            reduce_gaussians = config['gaussian_reduction']['reduce_gaussians'],
                                                                            reduction_type = config['gaussian_reduction']['reduction_type'],
                                                                            reduction_fraction=config['gaussian_reduction']['reduction_fraction'] )        
        
    params = initialize_xyzt(
            params,
            num_frames=num_frames,   # or config['dataset']['num_frames']
            xyzt_init_sigma=config['deforms'].get('xyzt_init_sigma', 5.0),
            device=params['means3D'].device
        )
    # local_means,local_rots,local_scales = deform_gaussians(params,0,False,5,'simple')
    # rendervar = transformed_GRNparams2rendervar(params,local_means,local_rots,local_scales)   

    # im,_,_ = Renderer(raster_settings=cam)(**rendervar)
    # plt.imshow(im.permute(1,2,0).cpu().detach()) 
    # plt.title('After initialize_first_timestep')           
    # plt.show()    
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = to01(tracking_color)                # fixes tracking path
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(), 
                                    use_simplification=config['gaussian_simplification'])
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    densification_frame_time_sum = 0
    densification_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]

            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                #color = color.permute(2, 0, 1) / 255
                color = to01(color)
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # timer.lap("all the config")
    added_new_gaussians = []





    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)): 
        if time_idx in dataset.eval_idx:
            with torch.no_grad():
            # Copy current (deformed) parameters to next time step
                # params['means3D'][..., time_idx+1] = params['means3D'][..., time_idx]
                # params['unnorm_rotations'][..., time_idx+1] = params['unnorm_rotations'][..., time_idx]
                # params['log_scales'][..., time_idx+1] = params['log_scales'][..., time_idx]
                # params['logit_opacities'][..., time_idx+1] = params['logit_opacities'][..., time_idx]
                # params['rgb_colors'][..., time_idx+1] = params['rgb_colors'][..., time_idx]
                params_iter = params[time_idx]

                # params[time_idx+1] = params[time_idx]
                params_iter['cam_unnorm_rots'][...,time_idx] = params_iter['cam_unnorm_rots'][...,time_idx-1]
                params_iter['cam_trans'][...,time_idx] = params_iter['cam_trans'][...,time_idx-1]
                try:
                    params[time_idx+1] = params_iter
                except:
                    params.append(params_iter)
                    print('Last frame is a test frame, appending to the list')
            continue    
        if isinstance(params,list):
            params_iter = params[time_idx]
            params_init = params[0]
        else:
            params_iter = params
            params_init = params
        iter_time_idx = time_idx
        motion_cfg = config.get('motion', {})
        motion_state = _ensure_motion_state(variables)
        base_gate_thresh = config['deforms'].get('xyzt_gate_thresh', 0.0)
        gate_thresh_tracking = _compute_gate_thresh(base_gate_thresh, motion_cfg, motion_state)
        maint_cfg = config.get('gaussian_maintenance', {})
        scene_state = variables.get('scene_state', {"mode": "slam"})
        maintenance_active_mask = None
        if maint_cfg.get('enable', False) and scene_state.get("mode", "slam") == "slam":
            with torch.no_grad():
                if maint_cfg.get('use_time_gate', True) and 't_mu' in params_iter:
                    gate_vals = xyzt_time_gate(params_iter, float(iter_time_idx))
                    gate_vals = gate_vals.reshape(-1)
                    slack = float(maint_cfg.get('active_gate_slack', 0.0))
                    effective_gate = max(0.0, gate_thresh_tracking - slack)
                    maintenance_active_mask = gate_vals > effective_gate
                else:
                    n_gauss = params_iter['logit_opacities'].shape[0]
                    maintenance_active_mask = torch.ones(n_gauss, device=params_iter['logit_opacities'].device, dtype=torch.bool)
            _update_opacity_and_visibility(params_iter, variables, time_idx, maint_cfg, maintenance_active_mask)
            _clamp_gaussian_params(params_iter, maint_cfg)
        # timer.lap("iterating over frame "+str(time_idx), 0)



        print() # always show global iteration
        # Load RGBD frames incrementally instead of all frames
        color, gt_depth, _, gt_pose = dataset[time_idx]

        # Predict depth using SurgeDepth
        if not config['depth']['use_gt_depth']:
            color_input = color.permute(2,0,1).unsqueeze(0).cuda()/255 # Change from WxHxC to BxCxWxH for inference
            color_input = torchvision.transforms.functional.normalize(color_input,config['depth']['normalization_means'],config['depth']['normalization_stds']) # Applying normalization
            pred_disp = model(color_input)
            t_pred = config['depth']['shift_pred']
            s_pred = config['depth']['scale_pred']
            t_gt =   config['depth']['shift_gt']
            s_gt =   config['depth']['scale_gt']
            pred_disp = ((model(color_input)-t_pred)/s_pred)*s_gt + t_gt #Fix this scaling offset
            depth = 1/pred_disp # Convert disp to depth
            depth = depth.permute(1,2,0) # CxWxH --> WxHxC to align with rest of the pipeline
            # plt.imshow(depth.squeeze().cpu().detach())
            # plt.show()
        else:
            depth = gt_depth

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        #color = color.permute(2, 0, 1) / 255
        color = to01(color)
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        # plt.imshow(depth.squeeze().cpu().detach())
        # plt.title('input depth')
        # plt.colorbar()
        # # plt.show()
        # plt.savefig(f'./scripts/plots/input_depth/{time_idx}.png')
        # plt.close()
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            #tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_color = to01(tracking_color)
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params_iter = initialize_camera_pose(
                params_iter,
                time_idx,
                forward_prop=config['tracking']['forward_prop'],
                motion_state=motion_state,
                motion_cfg=motion_cfg
            )

        # timer.lap("initialized data", 1)
        # If gaussians are randomly initialized, we need to optimize them enough to be able to perform tracking
        num_iters_initialization = config['GRN']['num_iters_initialization']
        if time_idx == 0:
            if config['GRN']['random_initialization']:        
                params_iter = optimize_initialization(
                    params_iter,
                    params_init,
                    curr_data,
                    num_iters_initialization,
                    variables,
                    iter_time_idx,
                    config,
                    gate_override=gate_thresh_tracking
                )
        # Tracking
        tracking_start_time = time.time()
        save_idx = 0
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            _set_requires_grad_for_phase(params_iter, config['tracking']['lrs'])  # <- add this
            optimizer = initialize_optimizer(params_iter, config['tracking']['lrs'])
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params_iter['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params_iter['cam_trans'][..., time_idx].detach().clone()
            if config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'gaussian':
                candidate_deform_biases = params_iter['deform_biases']
                candidate_deform_weights = params_iter['deform_weights']
                candidate_deform_stds = params_iter['deform_stds']
            elif config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'simple':
                candidate_means3D = params_iter['means3D'].detach().clone()
                candidate_unnorm_rots = params_iter['unnorm_rotations'].detach().clone()
                candidate_log_scales = params_iter['log_scales'].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            save_idx = 0
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params_iter,params_init, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter,use_gt_depth = config['depth']['use_gt_depth'],save_idx=None,gaussian_deformations=config['deforms']['use_deformations'],
                                                   use_grn = config['GRN']['use_grn'],deformation_type = config['deforms']['deform_type'], gate_thresh = gate_thresh_tracking)
                # print(loss)
                save_idx = save_idx+1

                # Backprop
                
                loss.backward()
                # Optimizer Update

                # weight_grad = params['deform_weights'].grad.mean()
                # bias_grad = params['deform_biases'].grad.mean()
                # stds_grad = params['deform_stds'].grad.mean()
                # cam_pos_grad = params['cam_trans'].grad.mean()
                # cam_rot_grad = params['cam_unnorm_rots'].grad.mean()
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    # params['deform_biases'].remove_hook()
                    # params['deform_weights'].remove_hook()
                    # params['deform_stds'].remove_hook()
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params_iter['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params_iter['cam_trans'][..., time_idx].detach().clone()
                        if config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'gaussian':
                            candidate_deform_biases = params_iter['deform_biases']
                            candidate_deform_weights = params_iter['deform_weights']
                            candidate_deform_stds = params_iter['deform_stds']
                        elif config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'simple':
                            candidate_means3D = params_iter['means3D'].detach().clone()
                            candidate_unnorm_rots = params_iter['unnorm_rotations'].detach().clone()
                            candidate_log_scales = params_iter['log_scales'].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params_iter, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params_iter['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params_iter['cam_trans'][..., time_idx] = candidate_cam_tran
                if config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'gaussian':
                    params_iter['deform_biases'] = candidate_deform_biases
                    params_iter['deform_weights'] = candidate_deform_weights
                    params_iter['deform_stds'] = candidate_deform_stds
                elif config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'simple':
                    params_iter['means3D'] = candidate_means3D
                    params_iter['unnorm_rotations'] = candidate_unnorm_rots
                    params_iter['log_scales'] = candidate_log_scales

        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1
        motion_state = _update_motion_state(variables, params_iter, time_idx, motion_cfg)
        gate_thresh_mapping = _compute_gate_thresh(base_gate_thresh, motion_cfg, motion_state)
        # we will fill this later after densify; init with 0
        added_this_frame = 0
        missing_ratio = None
        variables['added_this_frame'] = 0

        # plt.imshow(curr_data['depth'].permute(1,2,0).cpu().detach())
        # plt.colorbar()
        # plt.show()



        if not config['depth']['use_gt_depth']: # If we don't use gt depths, we still have to align predicted and rendered depth in scale
            invariant_depth = curr_data['depth']
            # plt.imshow(invariant_depth.squeeze().cpu().detach())
            if config['deforms']['use_deformations']:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(
                    params_iter, iter_time_idx, deform_grad=True, deformation_type=config['deforms']['deform_type']
                    )
            else:
                local_means = params_iter['means3D']
                local_rots = params_iter['unnorm_rotations']
                local_scales = params_iter['log_scales']
                local_opacities = params_iter['logit_opacities']
                local_colors = params_iter['rgb_colors']
            
            # local_means = params['means3D']
            # local_rots = params['unnorm_rotations']
            # local_scales = params['log_scales']
            transformed_pts = transform_to_frame(local_means,params_iter, time_idx, gaussians_grad=False, camera_grad=False)
            # img_rendervar = transformed_params2rendervar(params,transformed_pts,local_rots,local_scales)
            if config['GRN']['use_grn']:
                rendervar = transformed_GRNparams2depthplussilhouette(params_iter, curr_data['w2c'],
                                                                     transformed_pts,local_rots,local_scales,local_opacities)
            else:
                rendervar = transformed_params2depthplussilhouette(params_iter, curr_data['w2c'],
                                                                    transformed_pts,local_rots,local_scales,local_opacities)
            rendered_depth,_,_ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
            # im,_,_ = Renderer(raster_settings = curr_data['cam'])(**img_rendervar)
            mask = rendered_depth[1,:,:] > config['tracking']['sil_thres']

            # fig,ax = plt.subplots(2,4)
            # # ax[0,0].imshow(im.permute(1,2,0).cpu().detach())
            # # ax[0,0].set_title('Rendered im')
            # ax[0,1].imshow(curr_data['im'].permute(1,2,0).cpu().detach())
            # ax[0,1].set_title('Input img')
            # plt.imshow(rendered_depth[0,:,:].squeeze().cpu().detach())
            # plt.colorbar()
            # plt.show()

            # ax[1,0].set_title('Rendered depth')
            # ax0 = ax[1,1].imshow(curr_data['depth'].squeeze().cpu().detach())
            # ax[1,1].set_title('Input depth')
            # plt.colorbar(ax0,ax=ax[1,1])
            # # ax[0,2].imshow(nan_mask.squeeze().cpu().detach())
            # # ax[0,2].set_title('Nan mask')
            # # ax[0,3].imshow(bg_mask.squeeze().cpu().detach())
            # # ax[0,3].set_title('BG mask')
            # ax[1,2].imshow(mask.squeeze().cpu().detach())
            # ax[1,2].set_title('Mask')
            # plt.show()
            _,_,t_render, s_render, t_invar, s_invar = align_shift_and_scale(rendered_depth[0,:,:].unsqueeze(0),invariant_depth,mask.unsqueeze(0))
            curr_data['depth'] = ((invariant_depth-t_invar)/s_invar)*s_render + t_render
            curr_data['depth'] = curr_data['depth'].detach()
            # plt.imshow(curr_data['depth'].squeeze().cpu().detach())
            # plt.colorbar()
            # plt.show()
    
        if config['mapping']['perform_mapping']:
            # Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                # Densification
                if config['mapping']['add_new_gaussians'] and time_idx > 0:
                    # Setup Data for Densification
                    if seperate_densification_res:
                        # Load RGBD frames incrementally instead of all frames
                        densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                        #densify_color = densify_color.permute(2, 0, 1) / 255
                        densify_color = to01(densify_color)
                        densify_depth = densify_depth.permute(2, 0, 1)
                        densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                    'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                    else:
                        densify_curr_data = curr_data

                    # delete floating gaussians
                    # params, variables = remove_floating_gaussians(params, variables, densify_curr_data, time_idx)
                    densification_start_time = time.time()
                    # Add new Gaussians to the scene based on the Silhouette
                    pre_pts = params_iter['means3D'].shape[0]
                    scene_state_local = variables.get('scene_state', {"mode": "slam"})
                    in_protect = scene_state_local.get("mode", "slam") == "protect"
                    reduce_flag = False if in_protect else config['gaussian_reduction']['reduce_gaussians']
                    big_add_thresh_now = motion_cfg.get('big_add_thresh', 1500)
                    params_iter, variables, maybe_missing = add_new_gaussians(params_iter, variables, densify_curr_data, 
                                                        config['mapping']['sil_thres'], time_idx,
                                                        config['mean_sq_dist_method'], 
                                                        config['gaussian_simplification'],
                                                        nr_basis = config['deforms']['nr_basis'],
                                                        use_distributed_biases = config['deforms']['use_distributed_biases'],
                                                        total_timescale = config['deforms']['total_timescale'],
                                                        use_grn=config['GRN']['use_grn'],
                                                        grn_model=grn_model,
                                                        use_deform=config['deforms']['use_deformations'],
                                                        deformation_type=config['deforms']['deform_type'],
                                                        num_frames = num_frames,
                                                        random_initialization=config['GRN']['random_initialization'],
                                                        init_scale=config['GRN']['init_scale'],cam = cam,
                                                        reduce_gaussians = reduce_flag,
                                                        reduction_type = config['gaussian_reduction']['reduction_type'],
                                                        reduction_fraction=config['gaussian_reduction']['reduction_fraction'] )                     # if config['GRN']['random_initialization']:
                    post_pts = params_iter['means3D'].shape[0]
                    added_this_frame = max(0, post_pts - pre_pts)
                    missing_ratio = maybe_missing
                    if added_this_frame > big_add_thresh_now:
                        state_now = variables.get('scene_state', {"mode": "protect", "open_frames": 0, "recent_big_adds": 0})
                        state_now["mode"] = "protect"
                        state_now["open_frames"] = 0
                        state_now["recent_big_adds"] = 1
                        variables['scene_state'] = state_now
                    _clamp_gaussian_params(params_iter, maint_cfg)

                    added_new_gaussians.append(added_this_frame)
                    variables['added_this_frame'] = added_this_frame
                    densification_end_time = time.time()
                    densification_frame_time_sum += densification_end_time - densification_start_time
                    densification_frame_time_count += 1
                # if not config['distance_keyframe_selection']:
                #     with torch.no_grad():
                #         # Get the current estimated rotation & translation
                #         curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                #         curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                #         curr_w2c = torch.eye(4).cuda().float()
                #         curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                #         curr_w2c[:3, 3] = curr_cam_tran
                #         # Select Keyframes for Mapping
                #         num_keyframes = config['mapping_window_size']-2
                #         selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                #         selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                #         if len(keyframe_list) > 0:
                #             # Add last keyframe to the selected keyframes
                #             selected_time_idx.append(keyframe_list[-1]['id'])
                #             selected_keyframes.append(len(keyframe_list)-1)
                #         # Add current frame to the selected keyframes
                #         selected_time_idx.append(time_idx)
                #         selected_keyframes.append(-1)
                #         # Print the selected keyframes
                #         print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

                # # Reset Optimizer & Learning Rates for Full Map Optimization
                _set_requires_grad_for_phase(params_iter, config['mapping']['lrs'])
                optimizer = initialize_optimizer(params_iter, config['mapping']['lrs']) 
                if variables.get('scene_state', {"mode": "slam"}).get("mode", "slam") == "slam":
                    params_iter, variables = _cull_stale_gaussians(
                        params_iter, variables, optimizer, time_idx, maint_cfg, iter_time_idx, config
                    )
                _clamp_gaussian_params(params_iter, maint_cfg)

                with torch.no_grad():
                    if config['deforms']['use_deformations']:
                        local_means_tmp, _, _, _, _ = deform_gaussians(
                            params_iter, iter_time_idx, deform_grad=False, deformation_type=config['deforms']['deform_type']
                        )
                    else:
                        local_means_tmp = params_iter['means3D'].detach()

                    transformed_pts_tmp = transform_to_frame(
                        local_means_tmp, params_iter, iter_time_idx,
                        gaussians_grad=False, camera_grad=False
                    )
                    cam_z = transformed_pts_tmp[:, 2]
                    if 't_mu' in params_iter:
                        time_gate_tmp = xyzt_time_gate(params_iter, float(iter_time_idx))
                        active = (time_gate_tmp > gate_thresh_mapping).reshape(-1)
                    else:
                        active = torch.ones_like(cam_z, dtype=torch.bool)

                    near_thresh = 0.025
                    stale = (cam_z < near_thresh) & (~active)

                    counter = variables.get('front_gate_counter')
                    if counter is None or counter.shape[0] != cam_z.shape[0]:
                        counter = torch.zeros_like(cam_z, dtype=torch.int32, device=cam_z.device)
                    else:
                        counter = counter.clone()

                    counter[stale] += 1
                    counter[~stale] = 0
                    variables['front_gate_counter'] = counter

                    stale_mask = counter > 6
                    if stale_mask.any():
                        params_iter, variables = remove_points(stale_mask, params_iter, variables, optimizer)
                        torch.cuda.empty_cache()

                # # timer.lap("Densification Done at frame "+str(time_idx), 3)

                # # Mapping
                mapping_start_time = time.time()
                # if num_iters_mapping > 0:
                #     progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                    
                # actural_keyframe_ids = []
                # for iter in range(num_iters_mapping):
                #     iter_start_time = time.time()
                #     if not config['distance_keyframe_selection']:
                #         # Randomly select a frame until current time step amongst keyframes
                #         rand_idx = np.random.randint(0, len(selected_keyframes))
                #         selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                #         actural_keyframe_ids.append(selected_rand_keyframe_idx)
                #         if selected_rand_keyframe_idx == -1:
                #             # Use Current Frame Data
                #             iter_time_idx = time_idx
                #             iter_color = color
                #             iter_depth = depth
                #         else:
                #             # Use Keyframe Data
                #             iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                #             iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                #             iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                #     else:
                #         if len(actural_keyframe_ids) == 0:
                #             if len(keyframe_list) > 0:
                #                 curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                #                 actural_keyframe_ids = keyframe_selection_distance(time_idx, curr_position, keyframe_list, config['distance_current_frame_prob'], num_iters_mapping)
                #             else:
                #                 actural_keyframe_ids = [0] * num_iters_mapping
                #             print(f"\nUsed Frames for mapping at Frame {time_idx}: {[keyframe_list[i]['id'] if i != len(keyframe_list) else 'curr' for i in actural_keyframe_ids]}")

                #         selected_keyframe_ids = actural_keyframe_ids[iter]

                #         if selected_keyframe_ids == len(keyframe_list):
                #             # Use Current Frame Data
                #             iter_time_idx = time_idx
                #             iter_color = color
                #             iter_depth = depth
                #         else:
                #             # Use Keyframe Data
                #             iter_time_idx = keyframe_list[selected_keyframe_ids]['id']
                #             iter_color = keyframe_list[selected_keyframe_ids]['color']
                #             iter_depth = keyframe_list[selected_keyframe_ids]['depth']
                        
                        
                #     iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                #     iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                #                 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                #     # Loss for current frame
                #     loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                #                                     config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                #                                     config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],use_gt_depth = config['depth']['use_gt_depth'], mapping=True,save_idx = None,gaussian_deformations=config['deforms']['use_deformations'],
                #                                     use_grn = config['GRN']['use_grn'],deformation_type = config['deforms']['deform_type'])
                #     # Backprop
                #     loss.backward()
                #     with torch.no_grad():
                #         # Prune Gaussians
                if config['mapping']['prune_gaussians'] and variables.get('scene_state', {"mode": "slam"}).get("mode", "slam") == "slam":
                    pruning_cfg = config['mapping']['pruning_dict']
                    if motion_cfg.get('enable', False) and motion_state.get('is_contraction') and motion_cfg.get('pause_prune_override', False):
                        pruning_cfg = dict(pruning_cfg)
                        pruning_cfg['remove_big_after'] = min(pruning_cfg.get('remove_big_after', time_idx), time_idx)
                        pause_thresh = motion_cfg.get('pause_prune_thresh')
                        if pause_thresh is not None:
                            pruning_cfg['prune_size_thresh'] = min(pruning_cfg.get('prune_size_thresh', pause_thresh), pause_thresh)
                    params_iter, variables = prune_gaussians(params_iter, variables, optimizer, time_idx, pruning_cfg,config['GRN']['use_grn'])
                #         # Gaussian-Splatting's Gradient-based Densification
                #         if config['mapping']['use_gaussian_splatting_densification']:
                #             params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                #         # Optimizer Update
                #         optimizer.step()
                #         optimizer.zero_grad(set_to_none=True)
                #         # Report Progress
                #         if config['report_iter_progress']:
                #             report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                #                             mapping=True, online_time_idx=time_idx)
                #         else:
                #             progress_bar.update(1)
                #     # Update the runtime numbers
                #     iter_end_time = time.time()
                #     mapping_iter_time_sum += iter_end_time - iter_start_time
                #     mapping_iter_time_count += 1
                # if num_iters_mapping > 0:
                #     progress_bar.close()
                # # Update the runtime numbers
                num_iters_initialization = config['GRN']['num_iters_initialization_added_gaussians']

                big_add_limit = config.get('big_add_limit', motion_cfg.get('big_add_thresh', 1500))
                recent_added = variables.get('added_this_frame', added_this_frame)
                if recent_added > big_add_limit:
                    with torch.no_grad():
                        _clamp_gaussian_params(params_iter, maint_cfg)
                    print(f"[init] skipped optimize_initialization ({recent_added} new gaussians)")
                else:
                    params_iter = optimize_initialization(params_iter,params_init,curr_data,num_iters_initialization,variables,iter_time_idx,config, gate_override=gate_thresh_mapping)

                mapping_end_time = time.time()
                mapping_frame_time_sum += mapping_end_time - mapping_start_time
                mapping_frame_time_count += 1
                if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                    try:
                        # Report Mapping Progress
                        progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                        with torch.no_grad():
                            report_progress(params_iter, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                        progress_bar.close()
                    except:
                        ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                        save_params_ckpt(params_iter, ckpt_output_dir, time_idx)
                        print('Failed to evaluate trajectory.')
            
            # timer.lap('Mapping Done.', 4)

        variables['added_this_frame'] = added_this_frame
        scene_state = _update_scene_state(variables, motion_state, added_this_frame, missing_ratio, config)
        variables['scene_state'] = scene_state


        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params_iter['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params_iter['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        if time_idx < num_frames-1:
            with torch.no_grad():
            # Copy current (deformed) parameters to next time step
                # params['means3D'][..., time_idx+1] = params['means3D'][..., time_idx]
                # params['unnorm_rotations'][..., time_idx+1] = params['unnorm_rotations'][..., time_idx]
                # params['log_scales'][..., time_idx+1] = params['log_scales'][..., time_idx]
                # params['logit_opacities'][..., time_idx+1] = params['logit_opacities'][..., time_idx]
                # params['rgb_colors'][..., time_idx+1] = params['rgb_colors'][..., time_idx]
                params[time_idx+1] = params_iter

                # rendervar = 
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
    
        torch.cuda.empty_cache()
    nr_gauss = []
    for time_idx_plot in range(1, num_frames):
        if config['deforms']['use_deformations']:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                local_means, local_rots, local_scales, local_opacities, local_colors = deform_gaussians(
                params_iter, iter_time_idx, deform_grad=True, deformation_type=config['deforms']['deform_type']
                )
        else:
            local_means = params_iter['means3D']
            local_rots = params_iter['unnorm_rotations']
            local_scales = params_iter['log_scales']
            local_opacities = params_iter['logit_opacities']
            local_colors = params_iter['rgb_colors']
        
        nr_gauss.append(local_means.shape[0])
        # local_means = params['means3D']
        # local_rots = params['unnorm_rotations']
        # local_scales = params['log_scales']
        transformed_pts = transform_to_frame(local_means,params_iter, time_idx_plot, gaussians_grad=False, camera_grad=False)
        
        # img_rendervar = transformed_params2rendervar(params,transformed_pts,local_rots,local_scales)
        if config['GRN']['use_grn']:
            rendervar = transformed_GRNparams2rendervar(params[time_idx_plot], 
                                                                    transformed_pts,local_rots,local_scales,local_opacities,local_colors)
        else:
            rendervar = transformed_params2rendervar([time_idx_plot], 
                                                                transformed_pts,local_rots,local_scales,local_opacities,local_colors)
        im,_,_ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        ii = time_idx_plot
        img = Image.fromarray((im.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
        os.makedirs(f'./scripts/plots/final',exist_ok=True)
        img.save(f'./scripts/plots/final/{ii}.png')

    # timer.end()
    # plt.plot(nr_gauss)
    # plt.show()
    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        # mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    densification_frame_time_avg = densification_frame_time_sum / densification_frame_time_count
    
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print(f"Average Densification Time: {densification_frame_time_avg*1000} ms")
    with open(os.path.join(output_dir, "runtimes.txt"), "w") as f:
        f.write(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms\n")
        f.write(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s\n")
        f.write(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms\n")
        f.write(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s\n")
        f.write(f"Average Densification Time: {densification_frame_time_avg*1000} ms\n")
        f.write(f"Frame Time: {tracking_frame_time_avg + mapping_frame_time_avg+densification_frame_time_avg} s\n")
    

    if config['deforms']['deform_type'] == 'simple':
        params_save = {}
        # for time_idx in range(len(params)):
        #     if time_idx == 0:
        #         params_save['means3D'] = params[time_idx]['means3D'][:,:,None]
        #         params_save['unnorm_rotations'] = params[time_idx]['unnorm_rotations'][:,:,None]
        #         params_save['log_scales'] = params[time_idx]['log_scales'][:,:,None]
        #         params_save['logit_opacities'] = params[time_idx]['logit_opacities'][:,:,None]
        #         params_save['rgb_colors'] = params[time_idx]['rgb_colors'][:,:,None]
        #     else:
        #         params_save['means3D'] = torch.cat((params_save['means3D'], params[time_idx]['means3D'][:,:,None]), dim=2)
        #         params_save['unnorm_rotations'] = torch.cat((params_save['unnorm_rotations'], params[time_idx]['unnorm_rotations'][:,:,None]), dim=2)
        #         params_save['log_scales'] = torch.cat((params_save['log_scales'], params[time_idx]['log_scales'][:,:,None]), dim=2)
        #         params_save['logit_opacities'] = torch.cat((params_save['logit_opacities'], params[time_idx]['logit_opacities'][:,:,None]), dim=2)
        #         params_save['rgb_colors'] = torch.cat((params_save['rgb_colors'], params[time_idx]['rgb_colors'][:,:,None]), dim=2)
        params_save['means3D'] = [params[idx]['means3D'] for idx in range(len(params))]
        params_save['unnorm_rotations'] = [params[idx]['unnorm_rotations'] for idx in range(len(params))]
        params_save['log_scales'] = [params[idx]['log_scales'] for idx in range(len(params))]
        params_save['logit_opacities'] = [params[idx]['logit_opacities'] for idx in range(len(params))]
        params_save['rgb_colors'] = [params[idx]['rgb_colors'] for idx in range(len(params))]
        params_save['cam_unnorm_rots'] = params[time_idx]['cam_unnorm_rots']
        params_save['cam_trans'] = params[time_idx]['cam_trans']
        params = params_save

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    params_no_ints = {k: v for k, v in params.items() if not isinstance(k, int)}
    save_params(params_no_ints, output_dir)
    # save_means3D(params['means3D'], output_dir)

        # Evaluate Final Parameters
    dataset = [dataset, eval_dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(
            dataset,
            params,
            eval_dir,
            sil_thres=config['mapping']['sil_thres'],
            mapping_iters=config['mapping']['num_iters'],
            add_new_gaussians=config['mapping']['add_new_gaussians'],
            use_grn=config['GRN']['use_grn'],
            deformation_type=config['deforms']['deform_type'],
            gate_thresh=config['deforms'].get('xyzt_gate_thresh', 0.0),
            use_gt_depth=config['depth']['use_gt_depth'],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Prepare dir for visualization
    if args.online_vis:
        vis_dir = './online_vis'
        os.makedirs(vis_dir, exist_ok=True)
        for filename in os.listdir(vis_dir):
            os.unlink(os.path.join(vis_dir, filename))

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
    
    plot_video(os.path.join(results_dir, 'eval', 'plots'), os.path.join('./experiments/', experiment.group_name, experiment.scene_name, 'keyframes'))
