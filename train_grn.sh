#!/bin/bash
#SBATCH --nodes=1                                                                       # Specify the amount of A100 Nodes with 4 A100 GPUs (single GPU 128 SBUs/hour, 512 SBUs/hour for an entire node)
#SBATCH --ntasks=1                                                                      # Specify the number of tasks
#SBATCH --cpus-per-task=9                                                              # Specify the number of CPUs/task
#SBATCH --gpus=1                                                                        # Specify the number of GPUs to use
#SBATCH --partition=gpu_mig                                                                 # Specify the node partition (see slides Cris)
#SBATCH --time=00:10:00                                                                 # Specify the maximum time the job can run
#SBATCH --output=/gpfs/home6/hhuitema/github_repos/SurgeSplam/slurm_outputs/%j.out      # Define output folder for slurm output files



cd /gpfs/home6/hhuitema/github_repos/SurgeSplam



export WANDB_API_KEY=d7156e6b9496552b06075dfc9278a96a48a2a50e
export WANDB_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_CONFIG_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_CACHE_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_START_METHOD="thread"
wandb login

srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                 torchrun --nnodes 1 --nproc_per_node 1 train_GRN.py \
                                                                                           --data_path  /gpfs/work5/0/tesr0602/Datasets/SurgeNet \
                                                                                           --depth_path /scratch-shared/hhuitema/depths/Unfrozen_SurgeNet_Encoder \
                                                                                           --batch_size_per_gpu 1\
                                                                                           --num_workers 9\
                                                                                           --epochs 4 \
                                                                                           --output_dir /gpfs/home6/hhuitema/github_repos/SurgeSplam/logs/GRN_1 \
                                                                                           --learning_rate 1e-5 \
