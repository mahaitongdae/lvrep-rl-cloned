for ALG in sac; do
  for SIGMA in 0.0 1.0; do
    for SEED in 1; do
        # python main.py --alg $ALG --env Pendulum-v1 --sigma $SIGMA --max_timesteps 80000 --seed $SEED --hidden_dim 32
        python main.py --alg $ALG --env CartPendulum-v0 --sigma $SIGMA --max_timesteps 80000 --seed $SEED --hidden_dim 64
        # python main.py --no_reward_exponential --alg $ALG --env Pendubot-v0 --sigma $SIGMA --max_timesteps 150000 --seed $SEED --hidden_dim 128
        # python main.py --no_reward_exponential --alg $ALG --env Quadrotor2D-v2 --sigma $SIGMA --max_timesteps 150000 --seed $SEED --hidden_dim 128
    done
  done
done