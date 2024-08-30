#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GFSwget
#SBATCH --output=batch_out/BC_gfs_acq_%J_stdout.txt
#SBATCH --error=batch_out/BC_gfs_acq_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-5
#SBATCH --chdir=/home/bmac87/BoltCast/0_data_acquisition/
#SBATCH --time=48:00:00

conda init
conda activate wget

python wget_gfs.py --init_time 00 --fcst_hour $SLURM_ARRAY_TASK_ID --download
