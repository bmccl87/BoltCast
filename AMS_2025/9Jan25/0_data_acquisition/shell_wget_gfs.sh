#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GFSmv
#SBATCH --output=batch_out/BC_gfsmv%J_stdout.txt
#SBATCH --error=batch_out/BC_gfsmv_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/BoltCast/0_data_acquisition/
#SBATCH --time=48:00:00

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python shutil_mv_grib.py
