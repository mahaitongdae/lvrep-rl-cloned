#!/bin/bash
# (

#python main.py --alg sac --env Quadrotor2D-v2 --sigma 0.0 --seed 0

for ALG in rfsac; do
  for SIGMA in 0.0 1.0 2.0; do
    for MAX_TIMESTEPS in 150000; do
      for RF_NUM in 4096 2048; do
        for SEED in 0; do
          python main.py --use_random_feature --no_reward_exponential --critic_lr 3e-4 --alg $ALG --env Pendubot-v0 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --seed $SEED
        done
      done
    done
  done
done
#)


