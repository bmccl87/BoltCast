#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GLMs3fs
#SBATCH --output=batch_out/GLM_s3fs_%J_stdout.txt
#SBATCH --error=batch_out/GLM_s3fs_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/BoltCast/0_data_acquisition/
#SBATCH --array=4-6
#SBATCH --time=48:00:00

conda init
conda activate wget

python s3fs_glm.py --year=$SLURM_ARRAY_TASK_ID --sat=G18 --download
