import os
import numpy as np
import random
import torch


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        elif isinstance(v, list):
            res[k] = [x.detach().cpu().contiguous().numpy() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Pre-save finiteness validation: do not overwrite a good checkpoint with NaNs/Infs.
    nonfinite_info = {}
    for k, v in to_save.items():
        try:
            if isinstance(v, np.ndarray):
                # Count non-finite entries
                n_nonfinite = int((~np.isfinite(v)).sum())
                if n_nonfinite > 0:
                    nonfinite_info[k] = (n_nonfinite, v.shape)
        except Exception:
            # If any unexpected object, skip finiteness check for that key
            continue

    os.makedirs(output_dir, exist_ok=True)
    if len(nonfinite_info) > 0:
        # Save a diagnostic copy and warn instead of overwriting the main checkpoint.
        print(f"Warning: detected non-finite values in parameters; not saving main params.npz. Keys: {list(nonfinite_info.keys())}")
        diag_dir = os.path.join(output_dir, "bad_checkpoints")
        os.makedirs(diag_dir, exist_ok=True)
        bad_path = os.path.join(diag_dir, "params.bad.npz")
        # Save the bad checkpoint for debugging
        np.savez(bad_path, **to_save)
        # Also write a small textual report
        report_path = os.path.join(diag_dir, "params.bad.report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"Detected non-finite parameters when attempting to save to {output_dir}\n")
            for k, (cnt, shape) in nonfinite_info.items():
                rf.write(f"{k}: non-finite={cnt}, shape={shape}\n")
        return

    # Save the Parameters containing the Gaussian Trajectories
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_means3D(output_means, output_dir):
    # Save the Parameters containing the Gaussian means
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving means3D to: {output_dir}")
    save_path = os.path.join(output_dir, "means3D.ply")
    assert output_means.shape[1] == 3, "Tensor must be of shape (N, 3)"

    points = output_means.detach().cpu().numpy()

    ply_header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
""".format(len(points))

    with open(save_path, "w") as f:
        f.write(ply_header)
        np.savetxt(f, points, fmt="%f %f %f")


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Pre-save finiteness validation for checkpoints
    nonfinite_info = {}
    for k, v in to_save.items():
        try:
            if isinstance(v, np.ndarray):
                n_nonfinite = int((~np.isfinite(v)).sum())
                if n_nonfinite > 0:
                    nonfinite_info[k] = (n_nonfinite, v.shape)
        except Exception:
            continue

    os.makedirs(output_dir, exist_ok=True)
    if len(nonfinite_info) > 0:
        # Save a diagnostic copy and warn; include time_idx in filename for traceability
        print(f"Warning: detected non-finite values in checkpoint params at time {time_idx}; saving diagnostic copy instead of canonical checkpoint.")
        diag_dir = os.path.join(output_dir, "bad_checkpoints")
        os.makedirs(diag_dir, exist_ok=True)
        bad_path = os.path.join(diag_dir, "params"+str(time_idx)+".bad.npz")
        np.savez(bad_path, **to_save)
        report_path = os.path.join(diag_dir, "params"+str(time_idx)+".bad.report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"Detected non-finite parameters in checkpoint time {time_idx}\n")
            for k, (cnt, shape) in nonfinite_info.items():
                rf.write(f"{k}: non-finite={cnt}, shape={shape}\n")
        return

    # Save the Parameters containing the Gaussian Trajectories
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)