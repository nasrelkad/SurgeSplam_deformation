


cd /gpfs/home6/hhuitema/github_repos/SurgeSplam


srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                  python3  scripts/main_SurgeSplat.py configs/hamlyn/hamlyn_scene_01.py

