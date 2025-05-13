


cd /gpfs/home6/hhuitema/github_repos/SurgeSplam


srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                  python3  viz_scripts/final_recon.py 'experiments/EndoNerf cutting_deform_short_simple/cutting_deform_short_simple/config.py'


