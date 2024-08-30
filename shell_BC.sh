#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=batch_out/BC_train_%j_stdout.txt
#SBATCH --error=batch_out/BC_train_%j_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=BC_train
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/General_Exam/2_model_training/
#SBATCH --array=1-9

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up


. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0

python BC_train.py @txt_exp.txt @txt_unet.txt --exp=$SLURM_ARRAY_TASK_ID --cpus_per_task=16 --rotation=0
