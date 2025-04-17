cd /gpfs/home6/hhuitema/github_repos/SurgeSplam


srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                  ffmpeg -framerate 24 -i scripts/plots/tracking/rgb/%01d/24.png output.mp4