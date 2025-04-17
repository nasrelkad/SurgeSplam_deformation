import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

from GRN.datasets.SurgeNetStudent import concat_zip_datasets
from GRN.models.conv_unet import GaussianRegressionNetwork
from GRN.grn_train_utils import initialize_params, get_pointcloud#,transformed_GRNparams2depthplussilhouette,transformed_GRNparams2rendervar
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.recon_helpers import setup_camera,energy_mask

from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion,transformed_GRNparams2depthplussilhouette,transformed_GRNparams2rendervar
)
from utils.slam_external import calc_ssim


from pickle import dump


import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as F_trans
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import torchvision

import random

import wandb

import matplotlib.pyplot as plt
import GRN.distributed_utils as distributed_utils

import zipfile

def get_args_parser():
    parser = argparse.ArgumentParser('GRN',add_help=False)
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs of training.')


    parser.add_argument('--output_dir', default="logs/GRN_6", type=str, help='Path to save logs and checkpoints.')
    # parser.add_argument('--encoder', default = 'vitb', type = str, help = 'Encoder size of the modele')
    # parser.add_argument('--encoder_path',default='models/checkpoints/dynov2/modified_vitb14_dinov2_size336.pth',type = str, help ='path to the pretrained encoder weights')
    parser.add_argument('--batch_size_per_gpu', default = 1,type = int, help = 'batch size for each GPU')
    parser.add_argument('--num_workers', default = 2, type = int, help = 'number of data loading workers per GPU')
    parser.add_argument('--img_width', default = 336,type=int, help='input image width')
    parser.add_argument('--img_height', default=336,type = int, help='input image heigth')
    parser.add_argument('--data_path',default = '/media/thesis_ssd/data/SurgeNet_sample/', type = str, help = 'Path to data')
    parser.add_argument('--depth_path', default = '/media/thesis_ssd/data/SurgeNet_sample/Depths/SurgeNet_depths/',type = str, help = 'path to gt depth maps')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    # parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    # parser.add_argument('--freeze_encoder', default = True, type = bool, help='If true, the DinoV2 encoder weights are frozen and dont receive updates')
    parser.add_argument('--save_freq', default = 5, type = int, help = 'Frequency to save model at')
    # parser.add_argument('--data_split', default = 'test', type=str, help = 'Which data split to use for running the code, val contains only 100 samples, train contains all 100 000 images', choices = ['train','test','val'])
    parser.add_argument('--learning_rate',default = 5e-5,type = float, help = 'learning rate for optimizer')
    # parser.add_argument('--pretrained_learning_rate',default = 5e-6,type=float,help='learning rate for the pretrained encoder')
    parser.add_argument('--wandb_logging', default = True, type=bool, help = 'If true, enable weights and biases logging')
    parser.add_argument('--logging_interval',default = 10,type = int, help = 'Interval for wandb and terminal logging')
    # parser.add_argument('--teacher_path',default='/media/thesis_ssd/code/SurgeDepth/models/checkpoints/SurgeDepth_V6.pth', type = str,help = 'Path to the teacher model')
    parser.add_argument('--depth_loss_weight',default=0.002,type=float,help = 'Weighting factor for depth in loss function')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_accumulation_steps', default = 50, type = int, help = 'nr of batches to use for gradient accumultation')
    return parser


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None,GRN_input = False,plotting = False):
    global w2cs, w2ci
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)

    elif mapping:
        if do_ba: # Bundle Adjustment
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params['means3D'],params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)
    # print(transformed_pts.shape)
    # Initialize Render Variables
    if not GRN_input:
        rendervar = transformed_params2rendervar(params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_pts)
    else:
        # raise NotImplementedError('Not implemented yet')
        rendervar = transformed_GRNparams2rendervar(params,transformed_pts)
        depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, curr_data['w2c'],
                                                            transformed_pts)
    
    # Visualize the Rendered Images
    # online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False)
        
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification
    if plotting:
        plt.imshow(im.permute(1,2,0).cpu().detach())
        plt.show()
    # Depth & Silhouette Rendering
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)

    depth = depth_sil[0, :, :].unsqueeze(0)
    if plotting:
        plt.imshow(depth.squeeze(0).cpu().detach())
        plt.colorbar()
        plt.show()
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
    if plotting:
        plt.imshow(mask.squeeze(0).cpu().detach())
        plt.colorbar()
        plt.show()

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
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
    if plotting:
        plt.imshow(curr_data['im'].permute(1,2,0).cpu().detach())
        plt.show()
        plt.imshow(curr_data['depth'].permute(1,2,0).cpu().detach())
        plt.colorbar()
        plt.show()
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def train_surgedepth(args):
    
    ## Setup distributed training
    distributed_utils.init_distributed_mode(args)
    distributed_utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(distributed_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.wandb_logging:
        if args.rank == 0:
            wandb.init(
                project = 'GRN Training',

                config={
                    'learning_rate' : args.learning_rate,
                    'architecture': 'Conv- UNET',
                    'epochs': args.epochs,
                    'group': "DDM"
                }

            )

    normalize = transforms.Normalize([0.46888983, 0.29536288, 0.28712815],[0.24689102 ,0.21034359, 0.21188641])
    inv_normalize = transforms.Normalize([-0.46888983/0.24689102,-0.29536288/0.21034359,-0.28712815/0.21188641],[1/0.24689102,1/0.21034359,1/0.21188641]) #Take the inverse of the normalization

    ## Loading/preparing data
    transforms_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((args.img_height,args.img_width),antialias = False),
                                           normalize
    ])
    transform_depth = transforms.Compose([transforms.Resize((args.img_height,args.img_width),antialias = False)])

    datasets = ['Cholec80_video01_s0001.zip']

    dataset = concat_zip_datasets(args.data_path,args.depth_path,datasets=datasets,transform=transforms_train,depth_transform=transform_depth,train_student=False)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    dataset[0]
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler = sampler,
        batch_size = args.batch_size_per_gpu,
        num_workers = args.num_workers,
        pin_memory = False,
        drop_last = True
    )



    print(f"Data loaded: there are {len(dataset)} train images")
    ## Loading the Networks
    model = GaussianRegressionNetwork()   # TODO implement real model
    model.cuda()


    if distributed_utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Syncing batchnorms')



    ## Defining loss & Optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters())

    ## LR schedulers
    scheduler_depth = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)
    
    ## Option to resume training
    to_restore = {"epoch": 0}
    distributed_utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        criterion = criterion,
    )
    start_epoch = to_restore["epoch"]
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    # intrinsics =  torch.tensor([[300., 0., 16.], [0., 300., 16.], [0., 0., 1.]]).cuda()
    intrinsics = torch.tensor([[1.03530811e+03, 0., 5.96955017e+02], [0., 1.03508765e+03,5.20410034e+02], [0., 0., 1.]]).cuda() # SCARED (Da vinci Xi) intrinsics
    w2c = torch.eye(4).cuda()
    cam = setup_camera(336, 336, intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=False)
    render_params = {'intrinsics': intrinsics,
                     'w2c': w2c,
                     'cam': cam}

    ## Train loop
    for epoch in range(start_epoch,args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(args,model,data_loader=data_loader, optimizer=optimizer,criterion=criterion,epoch= epoch,render_params=render_params,inv_normalize = inv_normalize)

        save_dict = {'model': model.module.state_dict(),
                     'optimizer':optimizer.state_dict(),
                     'epoch': epoch+1,
                     'args': args,
                     'criterion': criterion.state_dict()
        }
               
        # val_one_epoch(model,data_loader,criterion,epoch)
        scheduler_depth.step()

        distributed_utils.save_on_master(save_dict,os.path.join(args.output_dir,'checkpoint.pth'))
        if epoch % args.save_freq ==0:
            distributed_utils.save_on_master(save_dict,os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        # INCLUDE LOGGING STEPS HERE
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch}
        torch.cuda.empty_cache()

    return None


def train_one_epoch(args,model,data_loader,optimizer,criterion,epoch,render_params,inv_normalize):
    model.train()
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    for it, data in enumerate(metric_logger.log_every(data_loader, args.logging_interval, header)):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATALOADING ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We need depth and images, can both be loaded with the same dataset as used for surgedepth
        # image = data[0].cuda() # BxCxWxH
        # depth = data[1].cuda()
        # # shift = torch.median(depth)
        # # scale = torch.mean(torch.abs(shift-depth))
        # scale = torch.max(depth)
        # depth = depth/scale
        # input = torch.cat([image,depth],1) # Concatenate along channel dimension, get Bx4xWxH
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setting up gaussians ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # pt_cloud,mean3_sq_dist= get_pointcloud(image.squeeze(0),depth.squeeze(0),render_params['intrinsics'],render_params['w2c'],compute_mean_sq_dist=True)
        # params,variables = initialize_params(pt_cloud,1,mean3_sq_dist)

        
        
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
        
        # gaussian_embedding = model(input) # Returns a Bx8x336x336 tensor
        # embed = torch.permute(gaussian_embedding[0], (1, 2, 0)).reshape(-1, 8) # (C, H, W) -> (H, W, C) -> (H * W, C)


        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prepare for rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # rots = embed[:,:4]   # Firsst 4 columns are rotation quaternioons
        # scales_norm = embed[:,4:7] # 3 columns for scales
        # opacities = embed[:,7][:,None] # 1 column for opacity

        # params['unnorm_rotations'] = rots
        # params['log_scales'] = scales_norm/scale
        # params['logit_opacities'] = opacities
        

        # rendervar = transformed_GRNparams2rendervar(params,params['means3D'])
        # depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, render_params['w2c'],
        #                                                 params['means3D'])
        
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ render gaussians ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # im_pred, radius, _ = Renderer(raster_settings=render_params['cam'])(**rendervar)
        # depth_pred,_,_ = Renderer(raster_settings=render_params['cam'])(**depth_sil_rendervar)



        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss+ Backwards pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # loss = criterion(im_pred, image.squeeze(0))+args.depth_loss_weight*criterion(depth_pred[0,:,:],depth.squeeze())

    
        # optimizer.zero_grad()
        # loss.backward()
        color = data[0].cuda().squeeze(0).detach()
        depth = 1/(data[1].cuda()+1e-4).squeeze(0).detach()
        # depth = data[1].cuda().squeeze(0)

        scale = torch.max(depth).detach()
        # scale = 1


        depth = (depth/scale).detach()
    
        input = torch.cat([color,depth],axis = 0).unsqueeze(0).cuda()
        # print(torch.max(depth))
        mask = depth > 0
        mask = mask.tile(1,3,1,1)
        mask = mask[0,0,:,:].reshape(-1)

        pt_cloud,mean3_sq_dist= get_pointcloud(inv_normalize(color),depth*scale,render_params['intrinsics'],render_params['w2c'],compute_mean_sq_dist=True,mask = mask)
        params,variables = initialize_params(pt_cloud,1,mean3_sq_dist)


        output = model(input)
        # print(output)
        cols = torch.permute(output[0], (1, 2, 0)).reshape(-1, 8) # (C, H, W) -> (H, W, C) -> (H * W, C)
        rots = cols[:,:4]

        scales_norm = cols[:,4:7]
        opacities = cols[:,7][:,None]

        params['unnorm_rotations'] = rots[mask]
        params['log_scales'] = scales_norm[mask]*mean3_sq_dist[:,None].tile(1,3)
        params['logit_opacities'] = opacities[mask]

 

        curr_data = {'cam': render_params['cam'], 'im': inv_normalize(color), 'depth': depth*scale, 'id': 0, 'intrinsics': render_params['intrinsics'], 
                        'w2c': render_params['w2c'], 'iter_gt_w2c_list': render_params['w2c']}

        # rendervar = transformed_GRNparams2rendervar(params,params['means3D'])
        # depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, curr_data['w2c'],
        #                                                 params['means3D'])
        # im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        # depth_pred,_,_ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)




        # mask_loss = depth>1
        # mask_loss = mask_loss
        # img_loss = 0.8*l1_loss_v1(im,color)+ 0.2*(1-calc_ssim(im,color))
        # depth_loss = torch.abs(curr_data['depth'] - depth_pred[0,:,:])[mask_loss].sum()/mask_loss.sum()
        # loss = img_loss+5*depth_loss
        # print(loss)
        loss_weights=dict(
            im=1.0,
            depth=0.002,
        )
        loss, _, _ = get_loss(params, curr_data, variables, 0, loss_weights,
                            use_sil_for_loss=True, sil_thres=0,
                            use_l1=True,ignore_outlier_depth_loss=True, tracking=False,mapping=True ,
                            plot_dir=None, visualize_tracking_loss=False,
                            tracking_iteration=0,GRN_input=True,plotting = False)

        loss = loss/args.num_accumulation_steps
        loss.backward()

        if ((it+1)%args.num_accumulation_steps == 0) or (it+1 ==x len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()



        # if not torch.isnan(loss):
        #     optimizer.step()
        # else:
        #     # Print some shite
        #     pass
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if it % args.logging_interval == 0:
        #     stats_log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        #     wandb.log(stats_log)

        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ SOME WANDB LOGGING ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with torch.no_grad():
            if it % args.logging_interval ==0: # Log every 10th batch
                if args.rank ==0:

                    rendervar = transformed_GRNparams2rendervar(params,params['means3D'])
                    depth_sil_rendervar = transformed_GRNparams2depthplussilhouette(params, render_params['w2c'],
                                                                    params['means3D'])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ render gaussians ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    im_pred, radius, _ = Renderer(raster_settings=render_params['cam'])(**rendervar)
                    depth_pred,_,_ = Renderer(raster_settings=render_params['cam'])(**depth_sil_rendervar)
                    fig,ax = plt.subplots(1,4)
                    ax[0].imshow(im_pred.permute(1,2,0).cpu().detach())
                    ax[0].set_title('Rendered Image')
                    ax[1].imshow(curr_data['im'].squeeze(0).permute(1,2,0).cpu().detach())
                    ax[1].set_title('Input Image')
                    ax[2].imshow(depth_pred[0,:,:].cpu().detach())
                    ax[2].set_title('Rendered Depth')
                    im = ax[3].imshow(depth.squeeze().cpu().detach())
                    ax[3].set_title('Input Depth')
                    plt.colorbar(im,ax = ax[3])

                    # Create images list for logging
                    images = [wandb.Image(fig, caption='Top: Gt disparity, bottom: predicted disparity', mode='L')]
                    stats_log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                    stats_log['images'] = images


                    wandb.log(stats_log)
                    plt.close()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # print(stats[0])
    return stats

def val_one_epoch(model,data_loader,criterion,epoch):
    with torch.no_grad():
        header = 'Validation epoch: [{}/{}]'.format(epoch, args.epochs)
        metric_logger_val = distributed_utils.MetricLogger(delimiter="  ")
        for it, data in enumerate(metric_logger_val.log_every(data_loader, 10, header)):
            data_student = data[0]
            data_alignment = data[1]

            img = data_student[0].cuda()
            disp = data_student[1].cuda().squeeze(1)
            img_alignment = data_alignment[0].cuda()



            mask = disp >0                       # Mask out any pixes with depth values of 0


            for i in range(disp.shape[0]): # According to DAv1 Paper, each depth map is normalized between 0 and 10
                disp[i,:,:] = (disp[i,:,:]-disp[i,:,:].min())/(disp[i,:,:].max()-disp[i,:,:].min())



            output = model.module(img).squeeze(1)              # Forward pass


            features_pretrained = frozen_encoder.get_intermediate_layers(img,12)
            features = model.module.pretrained.get_intermediate_layers(img_alignment,12)
            loss = criterion(output, disp,mask.squeeze(1)).item() + feature_loss(features_pretrained[0],features[0]).item()
            metric_logger_val.update(loss=loss)


        metric_logger_val.synchronize_between_processes()
        print("Averaged stats validation:", metric_logger_val)

    stats = {k: meter.global_avg for k, meter in metric_logger_val.meters.items()}
    if args.rank ==0:
        if args.wandb_logging:
            disp_viz = disp.clone() # Cloning the tensor to avoid inplace operations, avoids a memory leak
            output_viz = output.clone() # Cloning the tensor to avoid inplace operations, avoids a memory leak
            for i in range(5):
                # Normalize tensors (out-of-place operations)
                disp_viz[i,:,:] = (disp[i,:,:] - disp[i,:,:].min()) / (disp[i,:,:].max() - disp[i,:,:].min())
                output_viz[i,:,:] = (output[i,:,:] - output[i,:,:].min()) / (output[i,:,:].max() - output[i,:,:].min())

            # Ensure tensors are moved to CPU and converted to numpy
            disp_numpy = disp_viz[:5, :, :].cpu().detach().numpy()  # Move to CPU and then convert


            output_numpy = output_viz[:5, :, :].cpu().detach().numpy()  # Move to CPU and then convert

            # Concatenate and prepare for logging
            img_wandb = np.concatenate((disp_numpy, output_numpy), 1)  # Shape: 4x2HxW
            img_wandb = np.expand_dims(img_wandb, axis=3)
            img_wandb[img_wandb<0.00001] = 0.00001 # Need to clip images to be strictly within [0,1] or wandb outputs only black
            img_wandb[img_wandb>0.99999] = 0.99999 # Need to clip images to be strictly within [0,1] or wandb outputs only black

            # Create images list for logging
            images = [wandb.Image(img_wandb[i, :, :, :], caption='Top: Gt disparity, bottom: predicted disparity', mode='L') for i in range(5)]

            # Clear variables and free memory
            del disp, output, disp_numpy, output_numpy, disp_viz,output_viz,img_wandb
            torch.cuda.empty_cache()
    else:
        images = None
            
#   return stats,images



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SurgeDepth',parents = [get_args_parser()])
    args = parser.parse_args()
    train_surgedepth(args)