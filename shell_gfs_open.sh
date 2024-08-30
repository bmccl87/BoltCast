#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GFS_2_BC
#SBATCH --output=batch_out/GFS2BC_%J_stdout.txt
#SBATCH --error=batch_out/GFS2BC_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-16
#SBATCH --chdir=/home/bmac87/BoltCast/1_data_cleaning/
#SBATCH --time=48:00:00

conda init
conda activate pygrib

python gfs_2_BC.py --init_time 00 --fcst_hour $SLURM_ARRAY_TASK_ID
