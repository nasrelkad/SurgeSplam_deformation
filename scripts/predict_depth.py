#!/usr/bin/env python3
import argparse
import os
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets import (
    load_dataset_config,
    EndoSLAMDataset,
    C3VDDataset,
    ScaredDataset,
    EndoNerfDataset,
    RARPDataset,
    HamlynDataset,
    StereoMisDataset,
)

from models.SurgeDepth.dpt import SurgeDepth
from utils.eval_helpers import compute_errors
from utils.slam_helpers import align_shift_and_scale


def get_dataset(config_dict, basedir, sequence, **kwargs):
    name = config_dict["dataset_name"].lower()
    if name in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["scared"]:
        return ScaredDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["endonerf"]:
        return EndoNerfDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["rarp"]:
        return RARPDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["hamlyn"]:
        return HamlynDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["stereomis"]:
        return StereoMisDataset(config_dict, basedir, sequence, **kwargs)
    raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def load_config(config_path):
    loader = SourceFileLoader("depth_cfg", config_path)
    mod = loader.load_module()
    return mod.config


def setup_surge_depth(depth_cfg, device):
    if depth_cfg.get("use_gt_depth", False):
        raise ValueError("Depth prediction is disabled in this config (use_gt_depth=True).")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    cfg = model_configs[depth_cfg["model_size"]]
    model = SurgeDepth(**cfg).to(device)
    state = torch.load(depth_cfg["model_path"], map_location=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run SurgeDepth on GT RGB frames and compare against GT depth.")
    parser.add_argument("--config", required=True, help="Path to experiment config (e.g., configs/c3vdv2/c3vdv2_base.py)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit number of frames (default: all).")
    parser.add_argument("--pause", action="store_true", help="Wait for key press between frames.")
    parser.add_argument("--device", default="cuda", help="torch device to use.")
    parser.add_argument(
        "--viz-mode",
        choices=["aligned", "raw"],
        default="aligned",
        help="Choose whether to visualize raw depth or the aligned prediction used for metrics.",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=0,
        help="If >0, log shift/scale stats for the first N frames to help retune config depth parameters.",
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Skip GT depth usage; only visualize RGB vs predicted depth (no metrics).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    dataset_cfg = config["data"]
    def cfg_get(name, default=None):
        return dataset_cfg[name] if name in dataset_cfg else default
    gradslam_cfg = load_dataset_config(dataset_cfg["gradslam_data_cfg"])
    dataset = get_dataset(
        config_dict=gradslam_cfg,
        basedir=dataset_cfg["basedir"],
        sequence=os.path.basename(dataset_cfg["sequence"]),
        start=cfg_get("start", 0),
        end=cfg_get("end", -1),
        stride=cfg_get("stride", 1),
        desired_height=cfg_get("desired_image_height", None),
        desired_width=cfg_get("desired_image_width", None),
        device=device,
        relative_pose=True,
        ignore_bad=cfg_get("ignore_bad", False),
        use_train_split=cfg_get("use_train_split", False),
        train_or_test=cfg_get("train_or_test", "all"),
    )

    model = setup_surge_depth(config["depth"], device)
    norm_means = config["depth"]["normalization_means"]
    norm_stds = config["depth"]["normalization_stds"]
    t_pred = config["depth"]["shift_pred"]
    s_pred = config["depth"]["scale_pred"]
    t_gt = config["depth"]["shift_gt"]
    s_gt = config["depth"]["scale_gt"]

    num_frames = len(dataset) if hasattr(dataset, "__len__") else dataset.length
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    use_gt_depth = not args.visual_only
    if not use_gt_depth and args.viz_mode == "aligned":
        print("⚠️  Aligned visualization requires GT depth; falling back to raw predictions.")
        args.viz_mode = "raw"
    if not use_gt_depth and args.calib_frames > 0:
        print("⚠️  Calibration requires GT depth; ignoring --calib-frames.")

    all_errors = [] if use_gt_depth else None

    calib_stats = {"t_gt": [], "s_gt": [], "t_pred": [], "s_pred": []} if use_gt_depth else None

    with torch.no_grad():
        for idx in range(num_frames):
            color, gt_depth, _, _ = dataset[idx]

            color = color.to(device).float()
            gt_depth = gt_depth.to(device).float() if use_gt_depth else None

            color_input = color.permute(2, 0, 1).unsqueeze(0)  # BxCxHxW
            if color_input.max() > 1.5:
                color_input = color_input / 255.0
            color_input = torchvision.transforms.functional.normalize(color_input, norm_means, norm_stds)

            disp = model(color_input)
            disp = ((disp - t_pred) / s_pred) * s_gt + t_gt
            pred_depth = 1.0 / disp
            pred_depth = pred_depth.permute(1, 2, 0)  # HxWx1

            pred_depth_chw = pred_depth.permute(2, 0, 1)  # 1xHxW

            if use_gt_depth:
                gt_depth_chw = gt_depth.permute(2, 0, 1)
                valid_mask = (gt_depth_chw > 0).bool()
                gt_aligned, pred_aligned, t_gt_est, s_gt_est, t_pred_est, s_pred_est = align_shift_and_scale(
                    gt_depth_chw, pred_depth_chw, valid_mask
                )
                if args.calib_frames > 0 and idx < args.calib_frames:
                    calib_stats["t_gt"].append(t_gt_est.squeeze().cpu())
                    calib_stats["s_gt"].append(s_gt_est.squeeze().cpu())
                    calib_stats["t_pred"].append(t_pred_est.squeeze().cpu())
                    calib_stats["s_pred"].append(s_pred_est.squeeze().cpu())
                gt_valid = gt_aligned[valid_mask].cpu().numpy()
                pred_valid = pred_aligned[valid_mask].cpu().numpy()
                errors = compute_errors(gt_valid, pred_valid)
                all_errors.append(errors)

            color_np = color.detach().cpu().numpy()
            if color_np.max() <= 1.5:
                color_vis = np.clip(color_np * 255.0, 0, 255).astype(np.uint8)
            else:
                color_vis = np.clip(color_np, 0, 255).astype(np.uint8)
            if use_gt_depth:
                gt_depth_hw = gt_depth_chw.squeeze(0).detach().cpu().numpy()
            if args.viz_mode == "aligned" and use_gt_depth:
                pred_depth_hw = pred_aligned.squeeze(0).detach().cpu().numpy()
            else:
                pred_depth_hw = pred_depth_chw.squeeze(0).detach().cpu().numpy()

            if use_gt_depth:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(color_vis)
                axes[0].set_title(f"RGB Frame {idx}")
                axes[0].axis("off")

                im1 = axes[1].imshow(gt_depth_hw, cmap="viridis")
                axes[1].set_title("GT Depth")
                axes[1].axis("off")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                im2 = axes[2].imshow(pred_depth_hw, cmap="viridis")
                axes[2].set_title("Predicted Depth")
                axes[2].axis("off")
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                fig.suptitle(
                    f"Frame {idx} | AbsRel {errors[0]:.3f}, RMSE {errors[2]:.3f}, "
                    f"A1 {errors[4]:.3f}, A2 {errors[5]:.3f}"
                )
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(color_vis)
                axes[0].set_title(f"RGB Frame {idx}")
                axes[0].axis("off")

                im2 = axes[1].imshow(pred_depth_hw, cmap="viridis")
                axes[1].set_title("Predicted Depth")
                axes[1].axis("off")
                fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                fig.suptitle(
                    f"Frame {idx} | Pred depth range [{pred_depth_hw.min():.3f}, {pred_depth_hw.max():.3f}]"
                )

            plt.tight_layout()
            plt.show()
            if args.pause:
                input("Press Enter to continue...")

            if use_gt_depth:
                print(
                f"[{idx:04d}] AbsRel {errors[0]:.4f} | SqRel {errors[1]:.4f} | "
                f"RMSE {errors[2]:.4f} | RMSElog {errors[3]:.4f} | "
                f"A1 {errors[4]:.4f} | A2 {errors[5]:.4f} | A3 {errors[6]:.4f}"
            )

    if use_gt_depth and all_errors:
        metrics = np.array(all_errors)
        avg = metrics.mean(axis=0)
        header = "AbsRel SqRel RMSE RMSElog A1 A2 A3 PSNR"
        print("\nAverage metrics:")
        print(header)
        print(" ".join(f"{v:.6f}" for v in avg))

    if use_gt_depth and any(len(v) > 0 for v in calib_stats.values()):
        def summarize_calib(name, values):
            stacked = torch.stack(values)
            median = stacked.median().item()
            mean = stacked.mean().item()
            std = stacked.std(unbiased=False).item()
            print(f"  {name:>6}: median={median:.10f}, mean={mean:.10f}, std={std:.10f}")

        print("\nCalibration stats (copy medians into config['depth'] entries):")
        for key, values in calib_stats.items():
            if values:
                summarize_calib(key, values)
            else:
                print(f"  {key:>6}: not collected")


if __name__ == "__main__":
    main()
