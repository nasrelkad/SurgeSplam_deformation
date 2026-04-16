# Deformable Gaussian Splatting SLAM for Colonoscopy

## 🔥 Overview

This repository presents a **deformation-aware Gaussian Splatting SLAM framework** for monocular colonoscopy.
The goal is to **estimate camera pose and spatial coverage under strong non-rigid deformation**, enabling detection of *unobserved (blind-spot) regions* during endoscopic procedures.

Unlike prior work that focuses on reconstruction quality, this project identifies **camera pose estimation as the primary bottleneck** for reliable coverage reasoning in deformable environments.

---

## 🚀 Key Contributions

* **Pose-centric formulation** of deformable reconstruction
  → Shows that *pose observability*, not reconstruction fidelity, limits performance
* **Hybrid deformation model**
  → Combines graph-based, constant-velocity, and SVF motion
* **Temporal & confidence gating**
  → Stabilizes optimization under illumination changes and specular noise
* **Diagnostic framework for deformable GS-SLAM**
  → Systematic analysis of failure modes (topology change, depth instability, deformation drift)

---

## 🧠 Core Insight

> Reliable blind-spot detection in colonoscopy is fundamentally limited by **pose observability under deformation**, not by reconstruction quality alone.

---

## 🏗️ Method Overview

Pipeline:

1. Monocular depth prediction (SurgeDepth)
2. Gaussian initialization via RGB-D
3. Pose optimization via differentiable rendering
4. Hybrid deformation modeling
5. Temporal + confidence gating
6. Coverage reasoning via accumulated visibility

---

## 🎥 Results

### Static Reconstruction (C3VD)

* PSNR: **31.05 dB**
* SSIM: **0.89**
* Competitive with Gaussian Splatting baselines

### Pose Estimation

* ATE (rigid): **0.120 mm**
* ATE (deformable): **2.72 mm**

### Key Observations

* Stable pose → meaningful coverage gaps
* Unstable pose → false blind spots
* Deformation breaks geometric constraints → pose collapse

---

## ⚠️ Failure Modes

* **Topology change** → invalidates previous geometry
* **Depth inconsistency** → unstable initialization
* **Velocity accumulation** → exploding Gaussians
* **Fixed graph deformation** → stretching artifacts

---

## 📊 Benchmark Comparison

| Method            | PSNR      | SSIM     |
| ----------------- | --------- | -------- |
| PR-ENDO           | 34.24     | 0.90     |
| Gaussian Pancakes | 32.31     | 0.90     |
| **Ours (static)** | **31.05** | **0.89** |

---

## 🔬 Future Work

* Topology-aware deformation (dynamic graph re-binding)
* Global / batch SLAM optimization
* Illumination-aware rendering
* Learned deformation priors

---

## 📁 Project Structure

```
SurgeSplam_deformation/
│── configs/
│── datasets/
│── models/
│── scripts/
│── utils/
│── experiments/
│── outputs/
```

---

## 📌 Citation

If you use this work:

```
@mastersthesis{elkaddouri2026,
  title={Pose Prediction for Blind-Spot Detection in Deformable Colonoscopy},
  author={El Kaddouri, Nasr-Eddine},
  year={2026}
}
```

---

## 👤 Author

Nasr-Eddine El Kaddouri
MSc AI & Engineering Systems — TU Eindhoven

---
