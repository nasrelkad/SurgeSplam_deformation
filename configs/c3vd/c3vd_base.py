import os

scenes = [
    "cecum_t1_b", 
    "cecum_t2_b", 
    "cecum_t3_a", 
    "sigmoid_t1_a", 
    "sigmoid_t2_a", 
    "sigmoid_t3_a", 
    "trans_t1_b", 
    "trans_t2_c", 
    "trans_t4_a", 
    "trans_t4_b",
    "cecum_t4_b",
    "desc_t4_a",
    "trans_t1_a"
]

primary_device="cuda:0"
seed = 0
try:    
    scene_name = scenes[int(os.environ["SCENE_NUM"])]
except KeyError:
    scene_name = "trans_t1_a"

map_every = 1
keyframe_every = 8
# mapping_window_size = 24
tracking_iters = 25
mapping_iters = 50

group_name = "C3VD_base"
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
    mapping_window_size=24,#-1, # Mapping window size
    report_global_progress_every=2000, # Report Global Progress every nth frame
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=int(1e10), # Checkpoint Interval
    data=dict(
        basedir="./data/C3VD",
        gradslam_data_cfg="./configs/data/c3vd.yaml",
        sequence=scene_name,
        desired_image_height=1080//2,
        desired_image_width=1350//2,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        train_or_test="all",
    ),
    depth = dict(
        use_gt_depth = True,
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
        deform_type = 'simple',
        nr_basis = 50,
        use_distributed_biases = True,
        total_timescale = 50
    ),
    gaussian_reduction = dict(
        reduce_gaussians = True,
        reduction_type = 'laplace',
        reduction_fraction = 0.2
    ) ,  
    GRN = dict(
        use_grn = True,
        random_initialization = False,
        init_scale = -1.0,
        num_iters_initialization = 20,#10,
        num_iters_initialization_added_gaussians = 20,
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
        ),
        # grn_hidden_dim = 128,
        # grn_out_dim = 3,
        # grn_input_dim = 3,
        # grn_num_heads = 4,
        # grn_use_norm = True,
    ),
    tracking=dict(
        use_gt_poses=True, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.3, #0.5
            depth=1.50, #1.0
            deform = 0.5
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.005,
            
        ),
    ),
    mapping=dict(
        perform_mapping = True,
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.62,#0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=True,#False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=1.0,
            depth=0.75,#1.0,
            deform = 0.5
        ),
        lrs=dict(
            means3D=0.00005,#0.0001,
            rgb_colors=0.001,#0.0025,
            unnorm_rotations=0.0005,#0.001,
            logit_opacities=0.01,#0.05,
            log_scales=0.0005,#0.001,
            cam_unnorm_rots=0.000,
            cam_trans=0.000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=90,#20,
            prune_every=12,#20,
            removal_opacity_threshold=0.02,#0.005,
            final_removal_opacity_threshold=0.007,#0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10), # Doesn't consider iter 0
            prune_size_thresh = 0.1
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
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