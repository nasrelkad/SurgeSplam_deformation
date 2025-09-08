
#!/usr/bin/env python3

import os
import shutil
import argparse
import numpy as np
import trimesh
from trimesh.transformations import quaternion_from_matrix

def convert_surgesplam_to_gp(surge_root: str, gp_root: str):
    """
    Convert a SurgeSplam experiment folder into a GaussianPancakes dataset.

    Expects surge_root to contain:
      - params.npz           # numpy archive with 'means3D', 'intrinsics', 'w2c', 'org_width', 'org_height'
      - color/               # saved RGB PNGs
      - depth/               # saved depth PNGs
    """
    # 1) Prepare GP output structure
    img_dir = os.path.join(gp_root, "images")
    dep_dir = os.path.join(gp_root, "depths")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)

    # 2) Copy color and depth images
    for fname in sorted(os.listdir(os.path.join(surge_root, "color"))):
        shutil.copy(
            os.path.join(surge_root, "color", fname),
            os.path.join(img_dir, fname)
        )
        shutil.copy(
            os.path.join(surge_root, "depth", fname),
            os.path.join(dep_dir, fname)
        )

    # 3) Load parameters from params.npz
    params_path = os.path.join(surge_root, "params.npz")
    data = np.load(params_path)

    # Sparse Gaussian centers
    means = data['means3D']   # shape (N,3)

    # Camera intrinsics & image size
    fx, fy, cx, cy = data['intrinsics'][:4]
    W = int(data['org_width'])
    H = int(data['org_height'])

    # World->camera transforms
    w2c_all = data['w2c']    # shape (T,4,4)

    # 4) Write GP cameras.txt (PINHOLE)
    camera_file = os.path.join(gp_root, "cameras.txt")
    with open(camera_file, "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # 5) Convert and write camera_poses.txt in TUM format
    poses_out = os.path.join(gp_root, "camera_poses.txt")
    with open(poses_out, "w") as fout:
        for idx, M_w2c in enumerate(w2c_all):
            # invert to camera->world
            M_c2w = np.linalg.inv(M_w2c)
            t = M_c2w[:3, 3]
            qw, qx, qy, qz = quaternion_from_matrix(M_c2w)
            fout.write(
                f"{idx:0.6f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )

    # 6) Export sparse cloud as PLY
    ply_out = os.path.join(gp_root, "points3D.ply")
    cloud = trimesh.PointCloud(means)
    cloud.export(ply_out)

    print(f"[convert] GaussianPancakes dataset created at {gp_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SurgeSplam params.npz to GaussianPancakes dataset"
    )
    parser.add_argument(
        "--surge_root", required=True,
        help="SurgeSplam experiment folder (with params.npz, color/, depth/)"
    )
    parser.add_argument(
        "--gp_root", required=True,
        help="Output GaussianPancakes dataset root"
    )
    args = parser.parse_args()
    convert_surgesplam_to_gp(args.surge_root, args.gp_root)
