#!/bin/bash
#SBATCH --nodes=1                                                                       # Specify the amount of A100 Nodes with 4 A100 GPUs (single GPU 128 SBUs/hour, 512 SBUs/hour for an entire node)
#SBATCH --ntasks=1                                                                      # Specify the number of tasks
#SBATCH --cpus-per-task=9                                                              # Specify the number of CPUs/task
#SBATCH --gpus=1                                                                        # Specify the number of GPUs to use
#SBATCH --partition=gpu_mig                                                                 # Specify the node partition (see slides Cris)
#SBATCH --time=60:00:00                                                                 # Specify the maximum time the job can run
#SBATCH --output=/gpfs/home6/hhuitema/github_repos/SurgeSplam/slurm_outputs/%j.out      # Define output folder for slurm output files



cd /gpfs/home6/hhuitema/github_repos/SurgeSplam



export WANDB_API_KEY=d7156e6b9496552b06075dfc9278a96a48a2a50e
export WANDB_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_CONFIG_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_CACHE_DIR=/gpfs/home6/hhuitema/github_repos/SurgeSplam/wandb
export WANDB_START_METHOD="thread"
wandb login

srun apptainer exec --nv /home/hhuitema/docker/endogslam_v2.sif                 torchrun --nnodes 1 --nproc_per_node 1 train_GRN.py \
                                                                                           --data_path   \
                                                                                           --depth_path /scratch-shared/hhuitema/depths/Unfrozen_SurgeNet_Encoder \
                                                                                           --batch_size_per_gpu 1\
                                                                                           --num_workers 9\
                                                                                           --epochs 4 \
                                                                                           --output_dir /GRN/GRN_c3vd \
                                                                                           --learning_rate 1e-5 \


python scripts/train_GRN.py --depth_model_path /home/nasr/SurgeSplam/models/SurgeDepth/SurgeDepthStudent_V5.pth --output_dir /GRN/ --learning_rate 1e-5 --data_path  SurgeSplam/data/C3VD/sigmoid_t3_a/color \
--depth_path SurgeSplam/data/C3VD/sigmoid_t3_a/depth  --img_width 672 --img_height 532

    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')


    parser.add_argument('--output_dir', default="logs/GRN_8", type=str, help='Path to save logs and checkpoints.')
    # parser.add_argument('--encoder', default = 'vitb', type = str, help = 'Encoder size of the modele')
    # parser.add_argument('--encoder_path',default='models/checkpoints/dynov2/modified_vitb14_dinov2_size336.pth',type = str, help ='path to the pretrained encoder weights')
    parser.add_argument('--batch_size_per_gpu', default = 1,type = int, help = 'batch size for each GPU')
    parser.add_argument('--num_workers', default = 2, type = int, help = 'number of data loading workers per GPU')
    parser.add_argument('--img_width', default = 336,type=int, help='input image width')
    parser.add_argument('--img_height', default=336,type = int, help='input image heigth')
    parser.add_argument('--data_path',default = '/media/thesis_ssd/data/SurgeNet_sample/', type = str, help = 'Path to data')
    parser.add_argument('--depth_path', default = '/media/thesis_ssd/data/SurgeNet_sample/Depths/SurgeNet_depths/',type = str, help = 'path to gt depth maps')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    # parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    # parser.add_argument('--freeze_encoder', default = True, type = bool, help='If true, the DinoV2 encoder weights are frozen and dont receive updates')
    parser.add_argument('--save_freq', default = 5, type = int, help = 'Frequency to save model at')
    # parser.add_argument('--data_split', default = 'test', type=str, help = 'Which data split to use for running the code, val contains only 100 samples, train contains all 100 000 images', choices = ['train','test','val'])
    parser.add_argument('--learning_rate',default = 5e-5,type = float, help = 'learning rate for optimizer')
    # parser.add_argument('--pretrained_learning_rate',default = 5e-6,type=float,help='learning rate for the pretrained encoder')
    parser.add_argument('--wandb_logging', default = True, type=bool, help = 'If true, enable weights and biases logging')
    parser.add_argument('--logging_interval',default = 1000,type = int, help = 'Interval for wandb and terminal logging')
    # parser.add_argument('--teacher_path',default='/media/thesis_ssd/code/SurgeDepth/models/checkpoints/SurgeDepth_V6.pth', type = str,help = 'Path to the teacher model')
    parser.add_argument('--depth_loss_weight',default=0.002,type=float,help = 'Weighting factor for depth in loss function')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_accumulation_steps', default = 50, type = int, help = 'nr of batches to use for gradient accumultation')
    parser.add_argument('--masking_ratio', default = 0.8, type = float, help = 'Ratio of pixels to mask out for training')
    parser.add_argument('--depth_model_size', default = 'vitb', type = str, help = 'encoder size for depth estimation model')
    parser.add_argument('--depth_model_path', default = '/home/hhuitema/github_repos/SurgeSplam/models/SurgeDepth/SurgeDepthStudent_V5.pth', type = str, help = 'path to depth estimation checkpoint')