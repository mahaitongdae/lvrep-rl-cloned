#!/bin/bash

#
#for ALG in rfsac; do
#  for SIGMA in 0.0; do
#    for MAX_TIMESTEPS in 150000; do
#      for RF_NUM in 4096; do #4096
#        for NYSTROM_SAMPLE_DIM in 4096; do # currently not using top K
#          for SEED in 1 2 3; do # --critic_lr 1e-3
#            python main.py --use_nystrom --reward_exponential --critic_lr 1e-3 --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --seed $SEED
#          done
#        done
#      done
#    done
#  done
#done

for ALG in rfsac; do
  for SIGMA in 0.0; do
    for MAX_TIMESTEPS in 150000; do
      for RF_NUM in 4096; do #4096
#        for NYSTROM_SAMPLE_DIM in 8192; do # currently not using top K
        for SEED in 1; do # --critic_lr 1e-3
#          python main.py --use_random_feature --reward_exponential --critic_lr 1e-3 --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --seed $SEED
          python main.py --use_random_feature --reward_exponential --learn_rf --critic_lr 1e-3 --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --seed $SEED
          for R in 1.0 5.0 10.0; do
            python main.py --use_random_feature --robust_radius $R --reward_exponential --learn_rf --critic_lr 1e-3 --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --rf_num $RF_NUM --seed $SEED
          done
        done
      done
    done
  done
done