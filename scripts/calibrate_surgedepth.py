#!/usr/bin/env python3
"""
Utility script for inspecting SurgeDepth predictions and deriving safe
shift/scale values when ground-truth depth is unavailable.
"""

import argparse
import json
import os
import statistics
import sys
from importlib.machinery import SourceFileLoader
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torchvision

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets import (
    C3VDDataset,
    EndoNerfDataset,
    EndoSLAMDataset,
    HamlynDataset,
    RARPDataset,
    ScaredDataset,
    StereoMisDataset,
    load_dataset_config,
)
from models.SurgeDepth.dpt import SurgeDepth


def _load_experiment_config(path: str) -> Dict:
    module = SourceFileLoader("calibration_cfg", path).load_module()
    return module.config


def _get_dataset(config_dict, basedir, sequence, **kwargs):
    name = config_dict["dataset_name"].lower()
    if name in {"endoslam_unity"}:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"c3vd"}:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"scared"}:
        return ScaredDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"endonerf"}:
        return EndoNerfDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"rarp"}:
        return RARPDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"hamlyn"}:
        return HamlynDataset(config_dict, basedir, sequence, **kwargs)
    if name in {"stereomis"}:
        return StereoMisDataset(config_dict, basedir, sequence, **kwargs)
    raise ValueError(f"Unsupported dataset name {config_dict['dataset_name']}")


def _build_model(depth_cfg: Dict, device: torch.device) -> SurgeDepth:
    if depth_cfg.get("use_gt_depth", False):
        raise ValueError("SurgeDepth calibration is not required when use_gt_depth=True.")

    model_cfgs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    try:
        cfg = model_cfgs[depth_cfg["model_size"]]
    except KeyError as exc:
        raise ValueError(f"Unknown SurgeDepth variant '{depth_cfg['model_size']}'") from exc

    model = SurgeDepth(**cfg).to(device)
    state = torch.load(depth_cfg["model_path"], map_location=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _to_stats(tensor: torch.Tensor) -> Optional[Dict[str, float]]:
    if tensor is None:
        return None
    flat = tensor.reshape(-1)
    mask = torch.isfinite(flat)
    if mask.sum() == 0:
        return None
    flat = flat[mask].float()
    percentiles = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=flat.device)
    q = torch.quantile(flat, percentiles).cpu().tolist()
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "neg_frac": float((flat < 0).float().mean().item()),
        "near_zero_frac": float((flat.abs() < 1e-3).float().mean().item()),
    }


def _aggregate(stat_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    keys = ["min", "max", "mean", "median", "p01", "p05", "p50", "p95", "p99", "neg_frac", "near_zero_frac"]
    aggregated = {}
    collected = {k: [] for k in keys}
    for stats in stat_list:
        if not stats:
            continue
        for k in keys:
            if k in stats and np.isfinite(stats[k]):
                collected[k].append(stats[k])
    for k, values in collected.items():
        if values:
            aggregated[k] = float(statistics.median(values))
    return aggregated


def _suggest_shift_scale(percentiles: Dict[str, float]) -> Dict[str, float]:
    if not percentiles or "p50" not in percentiles:
        return {}
    median = percentiles["p50"]
    p05 = percentiles.get("p05", median)
    p95 = percentiles.get("p95", median)
    spread = max(p95 - p05, 1e-4)
    scale_pred = max(spread / 1.349, 1e-4)
    shift_pred = median
    target_shift = max(median, 1e-3)
    target_scale = max(scale_pred, 1e-4)
    return {
        "shift_pred": float(shift_pred),
        "scale_pred": float(scale_pred),
        "shift_gt": float(target_shift),
        "scale_gt": float(target_scale),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect SurgeDepth disparity/depth statistics for calibration.")
    parser.add_argument("--config", required=True, help="Experiment config (e.g. configs/c3vdv2/c3vdv2_base.py)")
    parser.add_argument(
        "--sequence",
        default=None,
        help="Override sequence name; defaults to the experiment config's data.sequence.",
    )
    parser.add_argument("--max-frames", type=int, default=10, help="Number of frames to analyse (default: 10).")
    parser.add_argument("--device", default="cuda", help="Torch device to run SurgeDepth on.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file capturing aggregated summary statistics.",
    )
    parser.add_argument(
        "--skip-config-calibration",
        action="store_true",
        help="Report stats on the raw model disparity without applying the config shift/scale.",
    )
    args = parser.parse_args()

    config = _load_experiment_config(args.config)
    depth_cfg = config["depth"]
    dataset_cfg = config["data"]

    device = torch.device(args.device)
    model = _build_model(depth_cfg, device)

    gradslam_cfg = load_dataset_config(dataset_cfg["gradslam_data_cfg"])
    sequence = args.sequence or os.path.basename(dataset_cfg["sequence"])

    dataset = _get_dataset(
        config_dict=gradslam_cfg,
        basedir=dataset_cfg["basedir"],
        sequence=sequence,
        start=dataset_cfg.get("start", 0),
        end=dataset_cfg.get("end", -1),
        stride=dataset_cfg.get("stride", 1),
        desired_height=dataset_cfg.get("desired_image_height", None),
        desired_width=dataset_cfg.get("desired_image_width", None),
        device=device,
        relative_pose=True,
        ignore_bad=dataset_cfg.get("ignore_bad", False),
        use_train_split=dataset_cfg.get("use_train_split", False),
        train_or_test=dataset_cfg.get("train_or_test", "all"),
    )

    norm_means = depth_cfg["normalization_means"]
    norm_stds = depth_cfg["normalization_stds"]
    t_pred = depth_cfg["shift_pred"]
    s_pred = depth_cfg["scale_pred"]
    t_gt = depth_cfg["shift_gt"]
    s_gt = depth_cfg["scale_gt"]

    frame_count = len(dataset)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    raw_disp_stats = []
    calibrated_disp_stats = []
    depth_stats = []

    with torch.no_grad():
        for idx in range(frame_count):
            color, _, _, _ = dataset[idx]
            color = color.to(device).float()
            color_input = color.permute(2, 0, 1).unsqueeze(0)
            if color_input.max() > 1.5:
                color_input = color_input / 255.0
            color_input = torchvision.transforms.functional.normalize(color_input, norm_means, norm_stds)

            disp_raw = model(color_input).squeeze(0)
            raw_disp_stats.append(_to_stats(disp_raw))

            if args.skip_config_calibration:
                disp_cal = disp_raw.clone()
            else:
                disp_cal = ((disp_raw - t_pred) / s_pred) * s_gt + t_gt
            calibrated_disp_stats.append(_to_stats(disp_cal))

            disp_safe = torch.clamp(disp_cal, min=1e-4)
            depth = 1.0 / disp_safe
            depth_stats.append(_to_stats(depth))

    agg_raw = _aggregate(raw_disp_stats)
    agg_cal = _aggregate(calibrated_disp_stats)
    agg_depth = _aggregate(depth_stats)
    suggestions = _suggest_shift_scale(agg_cal if agg_cal else agg_raw)

    print("=== SurgeDepth calibration summary ===")
    print(f"Sequence           : {sequence}")
    print(f"Frames analysed    : {frame_count}")
    print("\nRaw model disparity statistics (median of per-frame stats):")
    print(json.dumps(agg_raw, indent=2, sort_keys=True))
    print("\nConfigured disparity statistics (after applying current shift/scale):")
    print(json.dumps(agg_cal, indent=2, sort_keys=True))
    print("\nDerived depth statistics (1 / clamp(disp, 1e-4)):")
    print(json.dumps(agg_depth, indent=2, sort_keys=True))
    print("\nSuggested robust calibration (copy into config['depth']):")
    print(json.dumps(suggestions, indent=2, sort_keys=True))

    if args.output:
        payload = {
            "sequence": sequence,
            "frames": frame_count,
            "raw_disparity": agg_raw,
            "calibrated_disparity": agg_cal,
            "depth": agg_depth,
            "suggested_depth_config": suggestions,
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"\nSaved calibration summary to {args.output}")


if __name__ == "__main__":
    main()
