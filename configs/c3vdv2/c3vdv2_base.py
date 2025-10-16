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
tracking_iters = 12
mapping_iters = 50
frames = 100 # adjust untill which frame
group_name = "C3VDv2_base"
run_name = scene_name

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    distance_keyframe_selection=True, # Use Naive Keyframe Selection
    distance_current_frame_prob=0.1, # Probability of choosing the current frame in mapping optimization
    mapping_window_size=44, # Mapping window size
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
        desired_image_height=1080//2,
        desired_image_width=1350//2,
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
        deform_type = 'cv',
        nr_basis = 50,
        use_xyzt=True,
        xyzt_init_sigma=int(0.5*frames),
        use_distributed_biases = True,
        xyzt_gate_thresh=0.001,
        total_timescale = 50,
        max_vel_xyz=0.01,         # scene units / frame (tight)
        max_ang_vel=0.016,         # rad / frame (~8.6°/frame)
        max_logscale_vel=0.01,    # per frame
    ),
    gaussian_reduction = dict(
        reduce_gaussians = False,
        reduction_type = 'laplace',
        reduction_fraction = 0.05
    ) ,  
    GRN = dict(
        use_grn = True,
        random_initialization = True,
        init_scale = 0.02,
        num_iters_initialization = 10,
        num_iters_initialization_added_gaussians = 60,
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
            cv_vel_xyz = 0.0005,
            cv_vel_log_scales = 0.0002,
            cv_angvel_aa = 0.00005,
            feature_rest = 0.0,
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
            im=0.5,
            depth=1.0,
            deform = 0.0,
            cv_vel_l2=0.0001,  # NEW: tiny L2 priors on CV
            cv_ang_l2=0.0001,
            cv_scale_l2=0.0001,
            xyzt_compact=0.0001,
            xyzt_center=0.0001
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.005,
            deform_weights=0.001,
            deform_stds=0.0001,
            deform_biases=0.0001,
            cv_vel_xyz = 0.0003,
            cv_vel_log_scales = 0.00008,
            cv_angvel_aa = 0.0001,
            feature_rest = 0.00003,
            t_mu = 0.00001,
            t_logvar = 0.00001
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
            depth=1.0,
            deform = 0.3,
            cv_vel_l2=1e-4,  # same tiny priors
            cv_ang_l2=1e-4,
            cv_scale_l2=1e-4
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.002,
            unnorm_rotations=0.001,
            logit_opacities=0.005,
            log_scales=0.0005,
            cam_unnorm_rots=0.000,
            cam_trans=0.000,
            deform_weights=0.0005,
            deform_stds=0.0001,
            deform_biases=0.0001,
            cv_vel_xyz = 0.0003,
            cv_vel_log_scales = 0.0001,
            cv_angvel_aa = 0.0001,
            feature_rest = 0.001,
            t_mu = 0.00003,
            t_logvar = 0.00003
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=20,
            remove_big_after=100,
            stop_after=1000,
            prune_every=20,
            removal_opacity_threshold=0.01,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10), # Doesn't consider iter 0
            prune_size_thresh = 0.2
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
    ),
)