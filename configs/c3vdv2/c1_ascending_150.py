import os

# Config tuned for: data/C3VDv2/c1_ascending_t4_v4 (first 150 frames)
# Usage: python scripts/main_SurgeSplat.py configs/c3vdv2/c1_ascending_150.py

scenes = [
    "c1_ascending_t4_v4",
]

primary_device = "cuda:0"
seed = 0
try:
    scene_name = scenes[int(os.environ.get("SCENE_NUM", 0))]
except Exception:
    scene_name = scenes[0]

# Number of frames to consider for this run (first 150 frames)
frames = 150
group_name = "C3VDv2_c1_ascending_150"
run_name = f"{scene_name}_first{frames}"

# A slightly stronger mapping budget than small toy runs but still reasonable.
map_every = 1
keyframe_every = 8
tracking_iters = 48
mapping_iters = 60

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    distance_keyframe_selection=True,
    distance_current_frame_prob=0.25,
    mapping_window_size=-1,
    report_global_progress_every=2000,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True,
    checkpoint_interval=50,
    data=dict(
        basedir="./data/C3VDv2",
        gradslam_data_cfg="./configs/data/c3vdv2.yaml",
        sequence=scene_name,
        desired_image_height=540,  # matches preprocessing target (1080//2)
        desired_image_width=675,   # matches preprocessing target (1350//2)
        start=0,
        end=frames,
        stride=1,
        num_frames=frames,
        train_or_test="all",
    ),
    depth=dict(
        use_gt_depth=False,
        model_path='models/SurgeDepth/SurgeDepthStudent_V5.pth',
        model_size='vitb',
        normalization_means=[0.46888983, 0.29536288, 0.28712815],
        normalization_stds=[0.24689102, 0.21034359, 0.21188641],
        # Precomputed scaling factors (kept from base); fine-tune if you re-train SurgeDepth
        shift_pred=2.0192598978494627,
        scale_pred=0.5414197885483871,
        shift_gt=0.016469928791720198,
        scale_gt=0.0034374421235340256,
    ),
    deforms=dict(
        use_deformations=True,
        deform_type='gaussians',
        nr_basis=50,
        use_xyzt=True,
        graph=dict(
            num_nodes=256,
            K=6,
            sigma_mult=0.5,
            use_fourier=True,
            M=2,
        ),
        rebind_stride=5,
        rebind_tau_mult=2.5,
        xyzt_init_sigma=10,
        use_distributed_biases=True,
        xyzt_gate_thresh=0.17,
        total_timescale=frames,
        max_vel_xyz=0.05,
        max_ang_vel=0.6,
        max_logscale_vel=0.03,
        gates=dict(
            use_endo4dgs=True,
            endo4dgs=dict(normalize=True, eps=1e-6),
            use_ehsurgs=True,
            ehsurgs=dict(num_basis=6, init_sigma=12.0, init_bias=0.0, clamp_min=0.0, clamp_max=1.0, time_span=frames),
            prune_thresh=0.0,
        ),
        
    ),
    gaussian_reduction=dict(
        reduce_gaussians=False,
        reduction_type='laplace',
        reduction_fraction=0.2,
    ),
    GRN=dict(
        use_grn=True,
        random_initialization=False,
        init_scale=0.02,
        num_iters_initialization=12,
        num_iters_initialization_added_gaussians=60,
        sil_thres=0.05,
        model_path='models/GRN_v3.pth',
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.7,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        loss_weights=dict(im=0.5, depth=0.5, deform=0.0, arap=5e-3, node_vel_l2=1e-4, node_acc_l2=1e-5),
        lrs=dict(cam_unnorm_rots=0.002, cam_trans=0.005, deform_weights=0.001, deform_stds=0.0001, deform_biases=0.0001),
    ),
    mapping=dict(
        perform_mapping=True,
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.1,
        use_l1=True,
        use_sil_for_loss=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=1.0, depth=0.5, deform=0.0, arap=5e-3),
        lrs=dict(means3D=1e-4, rgb_colors=0.002, unnorm_rotations=0.001, logit_opacities=0.005, log_scales=0.0005),
        prune_gaussians=True,
        pruning_dict=dict(start_after=20, remove_big_after=100, stop_after=1000, prune_every=20, removal_opacity_threshold=0.01, final_removal_opacity_threshold=0.005, reset_opacities=False, reset_opacities_every=int(1e10), prune_size_thresh=0.2),
        use_gaussian_splatting_densification=True,
        densify_dict=dict(start_after=7, remove_big_after=80, stop_after=800, densify_every=15, grad_thresh=5e-5, num_to_split_into=2, removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005, reset_opacities_every=175),
    ),
    viz=dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=320,
        viz_h=320,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=30,
        enter_interactive_post_online=True,
        gaussian_simplification=False,
    ),
)
