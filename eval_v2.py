import pickle as pkl
import numpy as np
from envs.env_helper import *
import argparse
import os
from main import ENV_CONFIG
import torch
from agent.rfsac.rfsac_agent import RFVCritic, nystromVCritic
from agent.rfsac import rfsac_agent
from agent.sac import sac_agent
from agent.sac.actor import DiagGaussianActor
from utils.util import eval_policy

def eval(log_path, ):
    with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:        
        kwargs = pkl.load(f)
    env_name = kwargs['env']
    
    eval_config = ENV_CONFIG.copy()
    eval_config.update({'reward_scale': 1.,
                        'eval': True,
                        'reward_exponential': False,
                        'reward_type': 'energy',
                        'noise_scale': kwargs['sigma']})
    
    if env_name == "Pendulum-v1":
        eval_env = env_creator_pendulum(eval_config)
    elif env_name == 'Quadrotor2D-v2':
        eval_env = env_creator_quad2d(eval_config)
    elif env_name == 'Pendubot-v0':
        eval_env = env_creator_pendubot(eval_config)
    elif env_name == 'CartPoleContinuous-v0':
        eval_env = env_creator_cartpole(eval_config)
    eval_env = Gymnasium2GymWrapper(eval_env)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'rfsac':
        agent = rfsac_agent.RFSACAgent(**kwargs)

    use_nystrom = kwargs['use_nystrom']

    actor = DiagGaussianActor(obs_dim=kwargs['obs_space_dim'][0],
                              action_dim=kwargs['action_dim'],
                              hidden_dim=kwargs['hidden_dim'],
                              hidden_depth=2,
                              log_std_bounds=[-5., 2.])
    # if use_nystrom == False:
    #     critic = RFVCritic(**kwargs)
    # else:
    #     critic = nystromVCritic(**kwargs)

    actor.load_state_dict(torch.load(log_path+"/actor.pth"))
    # critic.load_state_dict(torch.load(log_path + "/critic.pth"))
    agent.actor = actor
    agent.device = torch.device("cpu")
    # agent.critic = critic

    eval_policy(agent, eval_env, eval_episodes=10)


if __name__ == '__main__':
    log_paths = {'Quadrotor':{},
                 'noisy_Quadrotor':{},
                 'Pendubot':{},
                 'noisy_Pendubot':{}}
    #### Quadrotor
    ## Nystrom
    log_paths['Quadrotor'].update({'nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/temp_good_results/rfsac_nystrom_True_rf_num_2048_sample_dim_8192/seed_0_2023-09-02-12-24-08',
                      })
    log_paths['noisy_Quadrotor']\
        .update({'random_feature':'/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/good_results/rfsac_nystrom_False_rf_num_4096/seed_0_2023-09-04-08-46-10',
                 'nystrom':'/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/good_results/rfsac_nystrom_True_rf_num_4096_sample_dim_8192/seed_2_2023-09-04-01-12-57'})

    # eval(log_paths['noisy_Quadrotor']['nystrom'])
    log_paths['Pendubot'].update({'nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_2048_sample_dim_2048/seed_0_2023-09-04-09-51-42'})
    log_paths['noisy_Pendubot'].update({'nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096/seed_0_2023-09-04-18-02-40'})
    eval(log_paths['noisy_Pendubot']['nystrom'])