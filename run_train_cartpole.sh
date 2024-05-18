#!/bin/bash
# (
#
#python main.py --alg sac --env Quadrotor2D-v2 --sigma 0.0 --seed 0

for ALG in rfsac; do
  for SIGMA in 0.0; do
    for RF_NUM in 8192; do
      for SEED in 1; do
        for LR in 3e-4; do
          python main.py --use_random_feature --critic_lr $LR --alg $ALG --env CartPendulum-v0 --sigma $SIGMA --max_timesteps 150000 --rf_num $RF_NUM --seed $SEED
          for R in 5.0 10.0 20.0; do
            python main.py --use_random_feature --robust_feature --robust_radius $R --critic_lr $LR --alg $ALG --env CartPendulum-v0 --sigma $SIGMA --max_timesteps 150000 --rf_num $RF_NUM --seed $SEED
          done
        done
      done
    done
  done
done
#)

#for ALG in rfsac; do
#  for SIGMA in 1.0; do
#    for RF_NUM in 8192; do #4096
#      for NYSTROM_SAMPLE_DIM in 8192; do # currently not using top K
#        for SEED in 1 2; do # --critic_lr 1e-3
#          python main.py --use_nystrom --critic_lr 3e-4 --alg $ALG --env CartPendulum-v0 --sigma $SIGMA --max_timesteps 200000 --rf_num $RF_NUM --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --seed $SEED
#        done
#      done
#    done
#  done
#done


