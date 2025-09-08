


cd /gpfs/home6/hhuitema/github_repos/SurgeSplam


srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                  python3  viz_scripts/offset_plots.py >> output.txt

