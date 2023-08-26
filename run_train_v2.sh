N=4
(
for ALG in rfsac; do
  for SIGMA in 0.0 1.0 2.0; do
    for MAX_TIMESTEPS in 100000; do
      for NYSTROM_SAMPLE_DIM in 512 1024 2048; do
        for SEED in 0 1 2 3 4; do
          for UES_NYSTROM in True False; do
            ((i=i%N)); ((i++==0)) && wait
            echo $i
            python main.py --alg $ALG --sigma $SIGMA --max_timesteps $MAX_TIMESTEPS --nystrom_sample_dim $NYSTROM_SAMPLE_DIM --seed $SEED --use_nystrom $UES_NYSTROM &
          done
        done
      done
    done
  done
done
)


