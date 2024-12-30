#!/bin/bash

for ALG in rfsac; do
  for SIGMA in 2.0 3.0; do
    for RF_NUM in 8192; do
      for LR in 3e-4; do
        for SEED in 1 2 3; do
          python main.py --use_random_feature --critic_lr $LR --alg $ALG --env Pendulum-v1 --sigma $SIGMA --max_timesteps 150000 --rf_num $RF_NUM --seed $SEED
        done
      done
    done
  done
done