#!/bin/bash
# (

#python main.py --alg sac --env Quadrotor2D-v2 --sigma 0.0 --seed 0

for ALG in rfsac; do
  for SIGMA in 0.0; do
    for MAX_TIMESTEPS in 150000; do
      for RF_NUM in 4096; do
        for SEED in 1; do
          python main.py --use_random_feature --no_reward_exponential --critic_lr 3e-4 --alg $ALG --env Pendubot-v0 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --seed $SEED
        done
      done
    done
  done
done
#)

for ALG in rfsac; do
  for SIGMA in 0.0 1.0; do
    for MAX_TIMESTEPS in 150000; do
      for RF_NUM in 4096; do #4096
        for NYSTROM_SAMPLE_DIM in 4096; do # currently not using top K
          for SEED in 1 2 3; do # --critic_lr 1e-3
            python main.py --use_nystrom --reward_exponential --learn_rf --critic_lr 1e-3 --alg $ALG --env Pendubot-v0 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --seed $SEED
          done
        done
      done
    done
  done
done


