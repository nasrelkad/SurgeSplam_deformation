import copy

from .c3vd_base import config as base_config

# Clone the base configuration so tweaks stay isolated from other scenes.
config = copy.deepcopy(base_config)

config['run_name'] = 'trans_t1_a'
config['workdir'] = './experiments/C3VD_trans_t1_a'
config['data']['sequence'] = 'trans_t1_a'

# Tune for 61 available frames (shorter temporal span than base defaults).
config['data']['end'] = 61
config['data']['num_frames'] = 61

# Match deformation helpers to the shortened sequence length.
config['deforms']['xyzt_init_sigma'] = 30
config['deforms']['total_timescale'] = 61

# Granular progress reports aligned with sequence size.
config['report_global_progress_every'] = 61

# Tailored optimization hyperparameters for the shorter clip.
config['tracking']['loss_weights'] = dict(
    im=0.75,
    depth=1.75,
    deform=0.0,
    cv_vel_l2=5e-5,
    cv_ang_l2=5e-5,
    cv_scale_l2=5e-5,
    xyzt_compact=5e-4,
    xyzt_center=5e-4,
)
config['tracking']['lrs'] = dict(
    means3D=0.0,
    rgb_colors=0.0,
    unnorm_rotations=0.0,
    logit_opacities=0.0,
    log_scales=0.0,
    cam_unnorm_rots=0.0015,
    cam_trans=0.0035,
    deform_weights=0.0,
    deform_stds=0.0,
    deform_biases=0.0,
    cv_vel_xyz=0.00025,
    cv_vel_log_scales=7.5e-05,
    cv_angvel_aa=8.0e-05,
    feature_rest=2.5e-05,
    t_mu=7.5e-06,
    t_logvar=7.5e-06,
)

config['mapping']['loss_weights'] = dict(
    im=1.5,
    depth=2.0,
    deform=0.1,
    cv_vel_l2=5e-5,
    cv_ang_l2=5e-5,
    cv_scale_l2=5e-5,
)
config['mapping']['lrs'] = dict(
    means3D=7.5e-05,
    rgb_colors=0.0015,
    unnorm_rotations=0.0008,
    logit_opacities=0.004,
    log_scales=0.0004,
    cam_unnorm_rots=0.0,
    cam_trans=0.0,
    deform_weights=0.0,
    deform_stds=0.0,
    deform_biases=0.0,
    cv_vel_xyz=0.00025,
    cv_vel_log_scales=8.5e-05,
    cv_angvel_aa=8.5e-05,
    feature_rest=8.0e-04,
    t_mu=2.5e-05,
    t_logvar=2.5e-05,
)
