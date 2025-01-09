#!/bin/bash
#
#SBATCH --partition=ai2es_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=batch_out/BCpermImp_%j_stdout.txt
#SBATCH --error=batch_out/BCpermImp_%j_stderr.txt
#SBATCH --job-name=BCpermImp
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-24%5
#SBATCH --chdir=/home/bmac87/BoltCast/3_model_analysis/
#SBATCH --time=12:00:00
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

features=("cape" "reflectivity" "precip_rate" "lifted_idx" "w" "ice_q" "rain_q" "snow_q" "graupel_q")
rotations=(4 3 2 1 0)

for rotation in "${rotations[@]}"; do
    for feature in "${features[@]}"; do
        echo ""
        echo ""
        echo "starting rotation $rotation and feature $feature"
        echo "---------------------------------------------------"
        echo "UNet"
        python 5_BC_perm_importance.py --rotation=$rotation --model_type=UNet --feature=$feature --perm_num=$SLURM_ARRAY_TASK_ID
        echo ""
        echo ""
        echo "starting rotation $rotation and feature $feature"
        echo "---------------------------------------------------"
        echo "LSTM"
        python 5_BC_perm_importance.py --rotation=$rotation --model_type=LSTM --feature=$feature --perm_num=$SLURM_ARRAY_TASK_ID
    done
done