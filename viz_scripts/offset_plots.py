import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import numpy as np
from utils.slam_helpers import transform_to_frame,transformed_params2depthplussilhouette,transformed_params2rendervar,transformed_GRNparams2rendervar,transformed_GRNparams2depthplussilhouette
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from scripts.main_SurgeSplat import deform_gaussians, setup_camera
import torch
from PIL import Image
from importlib.machinery import SourceFileLoader
import time
# import argparse



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



def generate_plots(args,scene_path):
    intrinsics = torch.tensor([[199.6883,   0.0000, 166.3290],
            [  0.0000, 249.4753, 170.4058],
            [  0.0000,   0.0000,   1.0000]], device='cuda:0')
    w2c = torch.tensor([[ 1.0000e+00,  6.5711e-11,  2.3283e-10,  0.0000e+00],
            [-3.1832e-11,  1.0000e+00, -7.4115e-21,  0.0000e+00],
            [-9.2644e-22,  2.9104e-11,  1.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device='cuda:0')

    cam = setup_camera(336,336, intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=True)


    # exp = 1
    print('Loading parameters from {}'.format(scene_path))
    start = time.time()
    params_np = np.load(scene_path,allow_pickle=True)
    print('Parameters loaded, loading took {} s'.format((start-time.time())*1000))
    params={}
    for key in params_np.keys():
        print("Loading {}".format(key))
        try:
            params[key] = torch.tensor(params_np[key]).cuda()
        except:
            params[key] = [torch.tensor(params_np[key][i]).cuda() for i in range(params_np[key].shape[0])]
    for i in range(params['cam_trans'].shape[-1]):
        params['cam_trans'][...,i][...,-1] += 0

    for id in range(params['cam_unnorm_rots'].shape[-1]):
        local_means,local_rots,local_scales,local_opacities,local_colors = deform_gaussians(params,id,deform_grad = True,deformation_type='simple')


        #  print(torch.sum(local_means-params['means3D']))

        transformed_pts = transform_to_frame(local_means,params,id,False,False)





        # Initialize Render Variables
        rendervar = transformed_GRNparams2rendervar(params, transformed_pts,local_rots,local_scales,local_opacities,local_colors)
        print(local_scales.max())
        rv_store = {}
        for key in rendervar.keys():
            rv_store[key] = rendervar[key].cpu().detach()
            local_means_store = local_means.cpu()
            local_scales_store = local_rots.cpu()
            local_rots_store = local_rots.cpu()
            transformed_pts_store = transformed_pts.cpu()



        #  rendervar['means3D'].retain_grad()

        depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, w2c,
                                                transformed_pts,local_rots,local_scales,local_opacities)


        #RGB Rendering

        rendervar['means2D'].retain_grad()
        im, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        # variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification
        img = Image.fromarray((im.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
        os.makedirs(args.output_path,exist_ok=True)
        # img.save(f'./eval_plots/plots_simple/{id}.png')
        img.save(os.path.join(output_path,f'{id}.png'))

        print(id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_config', default = './experiments/Hamlyn hamlyn_rectified01_1/hamlyn_rectified01_1/config.py')
    parser.add_argument('--output_path', default = './eval_plots/plots_simple/')

    args = parser.parse_args()
    
    experiment = SourceFileLoader(
        os.path.basename(args.experiment_config), args.experiment_config
    ).load_module()


    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]


    generate_plots(args,scene_path)
