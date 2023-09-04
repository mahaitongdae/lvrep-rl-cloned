# lvrep-rl modified by Zhaolin

To train an agent, use the main.py file (alternatively, the run_train.sh file). To evaluate an agent, use the eval.py file (alternatively, the run_eval.sh file).

The rfsac agent is the random fourier feature + SAC agent. The V critic network used by the rfsac agent is the RFVCritic critic in agent.rfsac.rfsac_agent file. The remaining elements of the agent (i.e. the actor) is an SAC actor.

TODOS:

- [ ] Pendubot Nystrom still use Layernorm, not compatible with Quadrotor2d.
- [ ] Quad2D RF uses slightly higher critic lr, 1e-3.

## Experimental Results:
### 2D Drones:
- Nystrom: /home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/temp_good_results/rfsac_nystrom_True_rf_num_2048_sample_dim_8192/seed_0_2023-09-02-12-24-08
  - top 2048 over 8192 sample dim
  - other can see temp good results dir
- RF: