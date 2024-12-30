#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p seas_compute
#SBATCH --mem=4G
#SBATCH -o ./output/pytorch_%j.out
#SBATCH -e ./output/pytorch_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=haitongma@g.harvard.edu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-3%3

# Load software modules and source conda environment
module load python/3.10.9-fasrc01
source activate torch39

# Run program
for ALG in sac; do
  for SIGMA in 0.0 1.0 2.0 3.0; do
    # for SEED in 1; do
        # python main.py --alg $ALG --env Pendulum-v1 --sigma $SIGMA --max_timesteps 80000 --seed $SEED --hidden_dim 32
        python main.py --alg $ALG --env Pendulum-v1 --sigma $SIGMA --max_timesteps 80000 --seed $SLURM_ARRAY_TASK_ID --hidden_dim 64
        # python main.py --no_reward_exponential --alg $ALG --env Pendubot-v0 --sigma $SIGMA --max_timesteps 150000 --seed $SEED --hidden_dim 128
        # python main.py --no_reward_exponential --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps 150000 --seed $SEED --hidden_dim 128
    # done
  done
done
# python main.py --alg sac --env Pendulum-v1 --sigma .0 --max_timesteps 80000 --hidden_dim 64 --seed $SLURM_ARRAY_TASK_ID
