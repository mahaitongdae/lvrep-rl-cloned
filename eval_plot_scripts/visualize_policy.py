import pickle as pkl
import numpy as np
from envs.env_helper import *
import argparse
import os
from main import ENV_CONFIG
import torch
from agent.rfsac.rfsac_agent import RFVCritic, nystromVCritic
from gymnasium.wrappers.record_video import RecordVideo
from agent.rfsac import rfsac_agent
from agent.sac import sac_agent
from agent.sac.actor import DiagGaussianActor
from utils.util import eval_policy, visualize_policy

def eval(log_path, ):
    with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:        
        kwargs = pkl.load(f)
    env_name = kwargs['env']
    
    eval_config = ENV_CONFIG.copy()
    eval_config.update({'reward_scale': 0.3,
                        'eval': True,
                        'reward_exponential': False,
                        'reward_type': 'energy',
                        'noise_scale': kwargs['sigma']})
    
    if env_name == "Pendulum-v1":
        env = gymnasium.make('Pendulum-v1', render_mode='rgb_array')
        eval_env = RescaleAction(env, min_action=-1., max_action=1.)
        eval_env = RecordVideo(eval_env, video_folder=log_path)
    elif env_name == 'Quadrotor2D-v2':
        eval_env = env_creator_quad2d(eval_config)
    elif env_name == 'Pendubot-v0':
        eval_env = env_creator_pendubot(eval_config)
    elif env_name == 'CartPoleContinuous-v0':
        eval_env = env_creator_cartpole(eval_config)
    elif env_name == 'CartPendulum-v0':
        eval_env = env_creator_cartpendulum(eval_config)
    # eval_env = Gymnasium2GymWrapper(eval_env)
    kwargs['action_space'] = eval_env.action_space
    # kwargs.update({'eval': True})
    # if kwargs['alg'] == "sac":
    #     agent = sac_agent.SACAgent(**kwargs)
    # elif kwargs['alg'] == 'rfsac':
    #     agent = rfsac_agent.RFSACAgent(**kwargs)

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

    actor.load_state_dict(torch.load(log_path+"/actor_last.pth"))
    # critic.load_state_dict(torch.load(log_path + "/critic.pth"))
    agent = rfsac_agent.DensityConstrainedLagrangianAgent(
        # state_dim=eval_env.observation_space.shape[0],
        #                                                   action_dim=eval_env.action_space.shape[0],
        #                                                   action_space=eval_env.action_space,
                                                          **kwargs)
    agent.actor = actor
    agent.device = torch.device("cpu")
    # agent.critic = critic

    ep_ret, ep_len = visualize_policy(agent, eval_env)

    return ep_ret


if __name__ == '__main__':
    log_paths = {'Quadrotor':{},
                 'noisy_Quadrotor':{},
                 'Pendubot':{},
                 'noisy_Pendubot':{},
                 'CartPoleContinuous':{},
                 'noisy_CartPoleContinuous':{},
                 'CartPendulum': {},
                 'noisy_CartPendulum':{}}

    eval('/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendulum-v1_sigma_0.0_rew_scale_1.0/density_nystrom_False_rf_num_512_learn_rf_False/try_evaluate_density/0/')