#!/bin/bash
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -t 0-01:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# Load software modules and source conda environment
module load python/3.10.9-fasrc01
source activate torch39

# Run program
srun -c 32 run_train_cluster.sh