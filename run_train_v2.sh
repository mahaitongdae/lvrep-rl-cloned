#N=4
#(
for ALG in rfsac; do
  for SIGMA in 0.0 1.0; do
    for MAX_TIMESTEPS in 400000; do
      for NYSTROM_SAMPLE_DIM in 1024; do #  2048 4096
        for RF_NUM in 1024 2048 4096; do
          for SEED in 0; do
            for ENV in Pendubot-v0; do
#              ((i=i%N)); ((i++==0)) && wait
#              echo $i
              python main.py --alg $ALG --env $ENV --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --rf_num $RF_NUM --seed $SEED
            done
          done
        done
      done
    done
  done
done
#)


