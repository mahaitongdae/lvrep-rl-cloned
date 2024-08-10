import pickle as pkl
import numpy as np
# from repr_control.envs.env_helper import *
from repr_control.envs import ArticulateParkingInfiniteHorizon
import argparse
import os
import torch
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.sac import sac_agent
from repr_control.agent.dpg import dpg_agent
from repr_control.agent.actor import DiagGaussianActor
from repr_control.utils.util import eval_policy
import gymnasium
import yaml

def eval(log_path, ):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)
    kwargs['device'] = 'cpu'

    # eval_env = gymnasium.make('ArticulateInfiniteHorizon-v0', render_mode='human', horizon=500, save_video=True)
    eval_env = ArticulateParkingInfiniteHorizon(render_mode='human', horizon=500, save_video=True)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'qpsac':
        agent = sac_agent.QPSACAgent(**kwargs)
    elif kwargs['alg'] == "mbdpg":
        agent = dpg_agent.DPGAgent(state_dim=6, action_dim=2, action_range = [[-1, -1], [1, 1]], **kwargs)
    else:
        raise NotImplementedError

    agent.actor.load_state_dict(torch.load(log_path+"/best_actor.pth"))
    # agent.actor = actor
    agent.device = torch.device("cpu")

    _, _, _, ep_rets = eval_biagt(agent, eval_env,
                                  eval_episodes=1,
                                  render=True,
                                  state=np.array([ 2.   ,       0.5    ,     -0.26179939 , 0.26179939 , 0.       ,   0.        ])
                                  ) # seed=3
                                  # seed=5)
    eval_env.close()

    return ep_rets

def eval_biagt(policy, eval_env, eval_episodes=100, render=False, seed=0, state=None):
    """
    Eval a policy
    """
    ep_rets = []
    avg_len = 0.
    for i in range(eval_episodes):
        ep_ret = 0.
        # eval_env.seed(i)
        if state is not None:
            state, _ = eval_env.reset(options={'state': state})
        else:
            state, _ = eval_env.reset(seed=seed) # options={"state": np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        done = False
        # print("eval_policy state", state)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
            avg_len += 1
            if render:
                eval_env.render()
        ep_rets.append(ep_ret)

    avg_ret = np.mean(ep_rets)
    std_ret = np.std(ep_rets)
    avg_len /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: avg eplen {avg_len}, avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}")
    print("---------------------------------------")
    return avg_len, avg_ret, std_ret, ep_rets


if __name__ == '__main__':
    eval("/Users/mahaitong/Code/repr_control/repr_control/log/mbdpg/parking/seed_0_2024-08-10-00-45-29")