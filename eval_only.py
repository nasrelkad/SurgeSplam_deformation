#!/usr/bin/env python3
"""
Quick script to re-evaluate saved parameters with fixed temporal gating.
"""
import sys
import os
import numpy as np
import torch
from utils.eval_helpers import eval_save

# Load config
config_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/C3VDv2_base/c1_transverse1_t4_v4/config.py"
import importlib.util
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

# Derive paths
exp_dir = config['workdir'] + "/" + config['run_name']
params_path = os.path.join(exp_dir, "params.npz")
eval_dir = os.path.join(exp_dir, "eval_fixed")
os.makedirs(eval_dir, exist_ok=True)

print(f"Loading parameters from: {params_path}")
params_np = np.load(params_path, allow_pickle=True)
params = {}
for key in params_np.files:
    if key in ('endo4dgs_net_state', 'endo4dgs_net_meta'):
        continue
    val = params_np[key]
    if isinstance(val, np.ndarray) and val.dtype == object:
        params[key] = val
    else:
        params[key] = torch.from_numpy(val).cuda().float()

# Load dataset
from datasets.gradslam_datasets import load_dataset_config
dataset_config = config["data"]
gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

if dataset_config["train_or_test"] == 'train':
    from datasets.gradslam_datasets import get_dataset
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=dataset_config["sequence"],
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device="cuda:0",
        relative_pose=True,
        train=False,
    )
    eval_dataset = dataset
    dataset = [dataset, eval_dataset, 'C3VD']
else:
    # Handle test split
    pass

print(f"Evaluating with fixed temporal gating...")
print(f"use_xyzt_gate = {config['deforms'].get('use_xyzt_gate', False)}")

# Evaluate
from utils.slam_helpers import rehydrate_endo4dgs
if config['deforms']['use_deformations'] and config['deforms']['deform_type'] == 'endo4dgs':
    params = rehydrate_endo4dgs(
        params,
        state=params_np.get('endo4dgs_net_state'),
        meta=params_np.get('endo4dgs_net_meta'),
        config=config['deforms'],
        device='cuda'
    )

log_scale_min = config['gaussian_maintenance']['log_scale_min']
log_scale_max = config['gaussian_maintenance']['log_scale_max']

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
        log_scale_bounds=(log_scale_min, log_scale_max),
        frustum_cull=True,
        depth_gate=False,
        use_xyzt_gate=config['deforms'].get('use_xyzt_gate', False),
    )

print(f"\nEvaluation complete! Results saved to: {eval_dir}")
