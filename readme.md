# SurgeSplam Deformation

Deformation-aware Gaussian Splatting experiments for endoscopic 3D reconstruction.

This repository is a research-oriented fork built around a **SurgeSplat / EndoGSLAM-style** pipeline, with code and utilities for:
- training and evaluating Gaussian-splatting-based reconstruction on endoscopic data,
- experimenting with **deformation modeling** and dynamic scene updates,
- converting outputs for downstream visualization and inspection,
- working with the **C3VD** dataset and related surgical / endoscopic reconstruction workflows.

> Status: experimental research code. Expect active development, dataset-specific assumptions, and environment setup work.

---

## Overview
<img width="514" height="202" alt="image" src="https://github.com/user-attachments/assets/8b7eaa84-7524-4680-8bde-bca3fc074a67" />

`SurgeSplam_deformation` extends a Gaussian-splatting reconstruction workflow toward **non-rigid / deformable scenes**, which is especially relevant for endoscopy and surgical video where tissue motion, camera motion, and partial observability happen at the same time.

The repository includes:
- training / reconstruction entry points,
- evaluation scripts,
- visualization helpers,
- configuration files for datasets such as **C3VD**,
- utilities for deformation modeling and export.

This makes it useful for:
- deformable 3D reconstruction research,
- Gaussian splatting experiments in medical imaging,
- testing reconstruction pipelines on endoscopic datasets,
- exporting splats to external viewers or downstream pipelines.

---

## Repository structure

```text
SurgeSplam_deformation/
├── .ipynb_notebooks/         # notebooks and experiments
├── GRN/                      # Gaussian / regression network related code
├── configs/                  # dataset and experiment configs
├── data/                     # dataset preprocessing / local dataset root
├── datasets/                 # dataset loaders
├── example_output/           # sample outputs
├── scripts/                  # main training / eval / conversion entry points
├── shell_run/                # helper shell scripts
├── slurm_outputs/            # cluster run outputs / logs
├── utils/                    # reconstruction, SLAM, rendering, and helper utilities
├── viz_scripts/              # visualization scripts
├── Dockerfile
├── requirements.txt
├── pose.txt
└── readme.md
```

### Important scripts

```text
scripts/
├── calc_metrics.py
├── convert_surgesplam_to_gp.py
├── main.py
├── main_SurgeSplat.py
├── main_SurgeSplat_gui.py
├── main_SurgeSplat_patched.py
├── main_full.py
├── test.py
└── train_GRN.py
```

A practical starting point is:
- `scripts/main_SurgeSplat.py` for the main reconstruction pipeline,
- `scripts/calc_metrics.py` for evaluation,
- `scripts/convert_surgesplam_to_gp.py` for export / conversion,
- `configs/c3vd/` for dataset-specific configuration.

---

## Features

- **Gaussian splatting reconstruction** for endoscopic / surgical scenes
- **Deformation-aware experimentation** for dynamic geometry
- **C3VD dataset support**
- Metrics / evaluation pipeline
- Export helpers for downstream visualization
- Notebook and visualization support for debugging and analysis

---

## Supported / intended datasets

The codebase is structured around endoscopic datasets and includes references to:
- **C3VD**
- other endoscopic / surgical datasets via dataset loaders in `datasets/`

The clearest out-of-the-box path appears to be **C3VD**.

---

## Installation

### 1) Create environment

```bash
conda create -n surgesplam python=3.10 -y
conda activate surgesplam
```

### 2) Install PyTorch

Install a PyTorch build that matches your CUDA version.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are on a different CUDA version, use the matching PyTorch index from the official install selector.

### 3) Install project dependencies

```bash
pip install -r requirements.txt
```

### 4) Build / install rasterization dependency

The repository depends on a depth-enabled Gaussian rasterizer:

```bash
pip install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/
```

> Note: compiling this package can require a working CUDA toolchain, compatible PyTorch, and a C++ compiler.

---

## Requirements

Main Python dependencies in `requirements.txt` include:

- `numpy==1.21.5`
- `Pillow==9.2.0`
- `opencv-python`
- `open3d==0.16.0`
- `kornia`
- `lpips`
- `torchmetrics`
- `pytorch-msssim`
- `trimesh`
- `cyclonedds`

Because this project mixes PyTorch, CUDA extensions, Open3D, and custom rasterization, it is best run in a **clean conda environment**.

---

## Data layout

Organize your dataset like this:

```text
data/
└── C3VD/
    ├── scene_name/
    │   ├── color/
    │   ├── depth/
    │   └── pose.txt
    └── another_scene/
```

Typical usage assumes:
- RGB frames in `color/`
- depth maps in `depth/`
- camera poses in `pose.txt`

Update the configuration files under `configs/` if your paths, image sizes, or camera intrinsics differ.

---

## Running reconstruction

### Default config-based run

```bash
python scripts/main_SurgeSplat.py configs/c3vd/c3vd_base.py
```

Depending on your experiment setup, you may also find these entry points useful:

```bash
python scripts/main.py configs/c3vd/c3vd_base.py
python scripts/main_full.py configs/c3vd/c3vd_base.py
```

---

## Evaluation

To evaluate a rendered scene against ground truth:

```bash
python scripts/calc_metrics.py \
  --gt data/C3VD/<scene_name> \
  --render experiments/C3VD_base/<scene_name> \
  --test_single
```

Replace `<scene_name>` with your actual sequence name.

---

## Export / conversion

To convert saved SurgeSplam outputs for other viewers or external pipelines, inspect:

```bash
python scripts/convert_surgesplam_to_gp.py
```

This script is useful when you want to:
- export splat parameters,
- inspect reconstruction outputs elsewhere,
- bridge this repository with other Gaussian-splatting tooling.

---

## Visualization

Useful places to inspect and visualize outputs:

- `example_output/`
- `viz_scripts/`
- `.ipynb_notebooks/`

These are good starting points for:
- debugging reconstruction quality,
- inspecting learned deformations,
- rendering point clouds / splats,
- comparing scene outputs.

---

## Typical workflow

1. Prepare dataset under `data/C3VD/...`
2. Choose or edit a config in `configs/c3vd/`
3. Run reconstruction with `main_SurgeSplat.py`
4. Inspect outputs in your experiment directory
5. Run `calc_metrics.py`
6. Use visualization / export scripts for analysis

---

## Notes on deformation experiments

This repository appears aimed at extending Gaussian splatting toward **dynamic or deformable anatomy**. In practice, that means you may need to tune:

- frame ranges,
- image resolution,
- camera intrinsics,
- deformation initialization,
- time-dependent parameters,
- export / viewer compatibility.

If you are using this repository for research, it is a good idea to document:
- the exact config used,
- dataset scene name,
- image resolution,
- CUDA / PyTorch versions,
- commit hash for reproducibility.

---

## Troubleshooting

### CUDA / extension build issues
If the rasterizer fails to build:
- verify `nvcc` is available,
- verify PyTorch CUDA matches your system CUDA,
- try a fresh environment,
- make sure a C++ compiler is installed.

### Open3D / GUI issues
Some Open3D visualizations can behave differently across Linux, Windows, and WSL. If you are on WSL, headless rendering or export-first workflows may be easier.

### Shape mismatch / dataset mismatch
If training or evaluation fails:
- verify the dataset folder layout,
- verify image resolution in the config,
- verify `pose.txt` matches the frame count,
- verify intrinsics in the dataset config.

---

## Reproducibility tips

For cleaner experiments:
- keep one config per scene,
- save logs per run,
- version experiment directories clearly,
- export final parameters after each successful run.

A common experiment structure is:

```text
experiments/
└── C3VD_base/
    └── <scene_name>/
        ├── params.npz
        ├── renders/
        ├── metrics/
        └── logs/
```

---

## Acknowledgments

This repository builds in the space of:
- **Gaussian Splatting / 3DGS**
- **SplaTAM**
- **EndoGSLAM / endoscopic Gaussian-splatting reconstruction**
- deformation-aware 3D reconstruction research

Please also credit upstream repositories and papers that this fork builds on.

---

## Citation

If you use this repository in academic work, cite:
1. this repository / fork,
2. the upstream repository it derives from,
3. the original Gaussian Splatting and related endoscopic reconstruction papers you use.

You may also want to add a project-specific BibTeX entry here once your method or results are finalized.

---

## Disclaimer

This is research code and may require adaptation for:
- new datasets,
- custom camera models,
- dynamic scene handling,
- production-level robustness.

---

## Contact

If you are maintaining this repository, consider adding:
- author name,
- email,
- project page,
- paper link,
- example results / screenshots,
- exact command lines for your best runs.

That will make the project much easier for others to reproduce and build upon.
