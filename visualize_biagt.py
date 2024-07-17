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
    eval_config.update({'reward_scale': 1.0,
                        'render': True,})
    
    eval_env = env_creator_articulate(eval_config)
    eval_env = Gymnasium2GymWrapper(eval_env)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'rfsac':
        agent = rfsac_agent.RFSACAgent(**kwargs)

    actor = DiagGaussianActor(obs_dim=kwargs['obs_space_dim'][0],
                              action_dim=kwargs['action_dim'],
                              hidden_dim=kwargs['hidden_dim'],
                              hidden_depth=kwargs['hidden_depth'],
                              log_std_bounds=[-5., 2.])

    actor.load_state_dict(torch.load(log_path+"/best_actor.pth"))
    agent.actor = actor
    agent.device = torch.device("cpu")

    _, _, _, ep_rets = eval_policy(agent, eval_env, eval_episodes=3, render=True, seed=2)

    return ep_rets


if __name__ == '__main__':
    eval("/home/haitong/PycharmProjects/lvrep-rl-cloned/log/Articulate-v0_sigma_1.0_rew_scale_1.0/sac/seed_0_2024-07-16-18-27-43")