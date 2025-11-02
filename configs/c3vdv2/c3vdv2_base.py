import os

scenes = [
    "c1_ascending_t4_v4", 
    "c1_transverse1_t4_v4", 
 
]

primary_device="cuda:0"
seed = 0
try:    
    scene_name = scenes[int(os.environ["SCENE_NUM"])]
except KeyError:
    scene_name = "c1_transverse1_t4_v4"

map_every = 1
keyframe_every = 8
# mapping_window_size = 24
tracking_iters = 10
mapping_iters = 20
frames = 50  # limit to first 50 frames of the sequence
group_name = "C3VDv2_base"
run_name = scene_name

config = dict(
    big_add_limit=1500,
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    distance_keyframe_selection=True, # Use Naive Keyframe Selection
    distance_current_frame_prob=0.1, # Probability of choosing the current frame in mapping optimization
    mapping_window_size=36, # Mapping window size tuned for 50-frame subset
    report_global_progress_every=2000, # Report Global Progress every nth frame
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=int(1e10), # Checkpoint Interval
    data=dict(
        basedir="./data/C3VDv2",
        gradslam_data_cfg="./configs/data/c3vdv2.yaml",
        sequence=scene_name,
        desired_image_height=532, #1080//2,
        desired_image_width=672 , #1350//2,
        start=0,
        end=frames,#-1,
        stride=1,
        num_frames=frames,#-1,
        train_or_test="all",
    ),
    depth = dict(
        use_gt_depth = False,
        model_path = 'models/SurgeDepth/SurgeDepthStudent_V5.pth',
        model_size = 'vitb',
        normalization_means = [0.46888983, 0.29536288, 0.28712815], 
        normalization_stds = [0.24689102 ,0.21034359, 0.21188641],
        shift_pred = 2.0192598978494627   ,
        scale_pred = 0.5414197885483871 ,
        shift_gt =   0.016469928791720198   ,
        scale_gt =   0.0034374421235340256    ,
    ), 
        deforms = dict(
            use_deformations = True,
            deform_type = 'deformgs',
            nr_basis = 32,              # leaner basis for 50-frame slice
            use_xyzt = True,
            xyzt_init_sigma = max(5, int(0.25 * frames)),
            use_distributed_biases = True,
            xyzt_gate_thresh = 0.35,
            total_timescale = frames,
            max_vel_xyz = 0.015,
            max_ang_vel = 0.025,
            max_logscale_vel = 0.015
    ),
    gaussian_reduction = dict(
        reduce_gaussians = True,
        reduction_type = 'laplace',
        reduction_fraction = 0.1
    ) ,  
    GRN = dict(
        use_grn = True,
        random_initialization = True,
        init_scale = 0.02,
        num_iters_initialization = 6,
        num_iters_initialization_added_gaussians = 30,
        sil_thres = 0.05,#0.0,
        model_path = 'models/GRN_v3.pth',
        random_initialization_lrs = dict(
            means3D=0.01,
            rgb_colors=0.001,
            unnorm_rotations=0.01,
            logit_opacities=0.001,
            log_scales=0.01,
            cam_unnorm_rots=0.000,
            cam_trans=0.0000,
            deform_weights=0.001,
            deform_stds=0.0001,
            deform_biases=0.0001,
            t_mu = 0.00005,
            t_logvar = 0.00005
        ),
        # grn_hidden_dim = 128,
        # grn_out_dim = 3,
        # grn_input_dim = 3,
        # grn_num_heads = 4,
        # grn_use_norm = True,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.95,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.6,
            depth=0.9,
            deform = 0.05,
            xyzt_compact=5e-5,
            xyzt_center=5e-5,
            inter_frame=0.0,
            inter_frame_rgb=0.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0015,
            cam_trans=0.0035,
            deform_weights=0.0008,
            deform_stds=0.00008,
            deform_biases=0.00008,
            t_mu = 5e-6,
            t_logvar = 5e-6
        ),
    ),
    mapping=dict(
        perform_mapping = True,
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.85, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=True,#False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=1.0,
            depth=0.8,
            deform = 0.4,
            inter_frame=0.05,
            inter_frame_rgb=0.1
        ),
        lrs=dict(
            means3D=8e-5,
            rgb_colors=0.0015,
            unnorm_rotations=0.0008,
            logit_opacities=0.004,
            log_scales=0.0004,
            cam_unnorm_rots=0.000,
            cam_trans=0.000,
            deform_weights=0.0004,
            deform_stds=8e-5,
            deform_biases=8e-5,
            t_mu = 2e-5,
            t_logvar = 2e-5
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=20,
            remove_big_after=20,
            stop_after=1000,
            prune_every=20,
            removal_opacity_threshold=0.015,
            final_removal_opacity_threshold=0.008,
            reset_opacities=False,
            reset_opacities_every=int(1e10), # Doesn't consider iter 0
            prune_size_thresh = 0.18
        ),
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=7,
            remove_big_after=80,
            stop_after=800,
            densify_every=15,
            grad_thresh=0.00005,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=175, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=320, viz_h=320,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=30, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        gaussian_simplification=False,
        viz_sil_thres=0.1,
        viz_depth_max=8.0,
    ),
    motion=dict(
        enable=True,
        contraction_speed_thresh=0.0015,  # meters per frame below which we consider the scope stationary
        min_pause_frames=3,               # number of consecutive low-motion frames to label a contraction
        direction_min_delta=0.003,        # ignore micro jitters when estimating forward direction
        gate_thresh_boost=0.1,            # tighten temporal gating while paused
        gate_thresh_cap=0.6,              # hard ceiling for the effective gate threshold
        forward_step_gain=1.05,           # extrapolate a bit beyond the last forward motion
        max_forward_step=0.04,            # clamp the extrapolated step to avoid jumps
        pause_forward_decay=0.4,          # decay the expected forward step during long pauses
        pause_prune_override=True,        # allow pruning to trigger even with few mapping steps
        pause_prune_thresh=0.16,          # temporary prune_size_thresh used while paused
        big_add_thresh=4000,              # gaussians added in a single frame to re-enter protect mode
        protect_open_frames=20,            # frames needed before switching back to slam
        min_forward_speed=5e-4           # reuseable override for scene-state forward detection
    ),
    gaussian_maintenance=dict(
        enable=True,
        opacity_decay=0.04,          # decay applied to logit opacity per frame since last observation
        opacity_floor=0.02,          # lower bound on opacity (in probability space)
        opacity_cap=0.98,            # optional ceiling to prevent over-saturation
        recency_boost=0.08,          # logit boost applied when a gaussian is observed
        visibility_prune_frames=25,  # prune gaussians unseen for this many frames
        visibility_min_hits=2,       # require at least this many hits before pruning for invisibility
        near_camera_thresh=0.012,    # remove gaussians that end up closer than this distance to the camera
        active_gate_slack=0.02,      # expand gate threshold when selecting active gaussians
        use_time_gate=True,          # rely on temporal gate to decide which gaussians were observed
        log_scale_min=-2.5,
        log_scale_max=2.5,
        logit_min=-6.0,
        logit_max=6.0
    ),
)
