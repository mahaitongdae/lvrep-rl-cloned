import pickle as pkl
import numpy as np
# from repr_control.envs.env_helper import *
from repr_control.envs import ArticulateParking
import argparse
import os
import torch
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.sac import sac_agent
from repr_control.agent.sac.actor import DiagGaussianActor
from repr_control.utils.util import eval_policy
import gymnasium

def eval(log_path, ):
    with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:        
        kwargs = pkl.load(f)
    env_name = kwargs['env']

    
    eval_env = gymnasium.make('ArticulateFiniteHorizon-v0', render_mode='human', horizon=250)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    else:
        raise NotImplementedError

    actor = DiagGaussianActor(obs_dim=len(eval_env.observation_space.low),
                              action_dim=kwargs['action_dim'],
                              hidden_dim=kwargs['hidden_dim'],
                              hidden_depth=kwargs['hidden_depth'],
                              log_std_bounds=[-5., 2.])

    actor.load_state_dict(torch.load(log_path+"/best_actor.pth"))
    agent.actor = actor
    agent.device = torch.device("cpu")

    _, _, _, ep_rets = eval_policy(agent, eval_env, eval_episodes=10, render=True, seed=2)

    return ep_rets


if __name__ == '__main__':
    eval("/home/haitong/PycharmProjects/lvrep-rl-cloned-toolbox/repr_control/log/sac/parking/seed_0_2024-07-19-00-11-29")