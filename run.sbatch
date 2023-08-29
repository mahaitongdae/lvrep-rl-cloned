#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-01:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# Load software modules and source conda environment
module load python/3.10.9-fasrc01
source activate torch39

# Run program
srun run_train_v2.sh