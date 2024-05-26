# Stochastic Nonlinear Control via Finite-dimensional Spectral Dynamic Embedding

This is the implementation of [Stochastic nonlinear control via finite-dimensional spectral dynamic embedding](https://arxiv.org/abs/2304.03907). A short version of this paper has shown up in [CDC 2023](https://ieeexplore.ieee.org/abstract/document/10383842).

## Installation
```bash
# in your virtual/conda environments
pip install -r requirements.txt
```

## Sample scripts of running experiments
- Random feature
  ```bash
  python main.py --use_random_feature --no_reward_exponential --critic_lr 3e-4 --alg rfsac --env Pendubot-v0 --sigma 1.0 --max_timesteps 150000 --rf_num 8192 --seed 0
  ```
- Nystrom feature
  ```bash
  python main.py --use_nystrom --no_reward_exponential --critic_lr 3e-4 --alg rfsac --env Pendubot-v0 --sigma 1.0 --max_timesteps 150000 --rf_num 8192 --nystrom_sample_dim 2048 --seed 0
  ```
  Please refer to `run_train_$ENV.sh` for more training scripts and recommended hyperparameters.

## Baselines

The code for iLQR is found in the branch iLQC. Please use test_main.py in that branch to train and evaluate iLQR on the pendulum swingup problem.

The code for Koopman control is found in the branch DeepKoopman. Please use the Learn_Koopman_with_KlinearEig.py file in the train folder of that branch to train a new Koopman dynamics model; for evaluation, please use the og_eval_mpc.py file in the control folder of that branch.

## Citations
```
@article{ren2023stochastic,
      title={Stochastic Nonlinear Control via Finite-dimensional Spectral Dynamic Embedding}, 
      author={Tongzheng Ren and Zhaolin Ren and Haitong Ma and Na Li and Bo Dai},
      year={2023},
      eprint={2304.03907},
      archivePrefix={arXiv}
}
```
