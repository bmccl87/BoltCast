#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=G17BCDay
#SBATCH --output=batch_out/G17BCDay_%J_stdout.txt
#SBATCH --error=batch_out/G17BCDay_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/BoltCast/1_data_cleaning/
#SBATCH --time=48:00:00

conda init
conda activate pygrib

python glm_open_day_BC.py 