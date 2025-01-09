#!/bin/bash
#
#SBATCH --partition=ai2es_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=batch_out/BC3DUNet_%j_stdout.txt
#SBATCH --error=batch_out/BC3DUNet_%j_stderr.txt
#SBATCH --job-name=BC3DUNet
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0
#SBATCH --chdir=/home/bmac87/BoltCast/2_model_training/UNet3D/
#SBATCH --time=08:00:00
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python BC_train_unet.py --rotation=$SLURM_ARRAY_TASK_ID @txt_exp.txt @txt_proj.txt @txt_unet.txt
