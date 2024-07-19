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

    _, _, _, ep_rets = eval_biagt(agent, eval_env, eval_episodes=1, render=True, seed=2)

    return ep_rets

def eval_biagt(policy, eval_env, eval_episodes=100, render=False, seed=0):
	"""
	Eval a policy
	"""
	ep_rets = []
	avg_len = 0.
	for i in range(eval_episodes):
		ep_ret = 0.
		# eval_env.seed(i)
		state, _ = eval_env.reset(options={"state": np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
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
    eval("/homes/haitongma/PycharmProjects/repr_control/repr_control/log/sac/parking/seed_0_2024-07-19-14-41-00")