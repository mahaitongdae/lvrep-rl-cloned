import time

import numpy as np
import torch
import gym
import argparse
import os

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent

from our_env.noisy_pend import noisyPendulumEnv

from agent.rfsac.rfsac_agent import RFVCritic
from agent.sac.actor import DiagGaussianActor
from agent.rfsac.rfsac_agent import RFSACAgent

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from copy import deepcopy
import time


def main(env, log_path, agent, rf_num, learn_rf, max_steps=200, state_dim=None):
    np.random.seed(0)
    # n_init_states = 100
    # low_arr =  [-np.pi,-1.]
    # high_arr = [np.pi, 1.]
    # init_states = np.random.uniform(low = low_arr, high = high_arr, size = (n_init_states,state_dim))
    n_init_states = 1
    init_states = [np.array([-0.5, 0])]
    all_rewards = np.empty(n_init_states)
    u_list = np.empty(max_steps)
    for i in np.arange(n_init_states):
        init_state = init_states[i]
        state = env.reset(init_state=init_state) if args.env == 'Pendulum-v1' else env.reset()
        eps_reward = 0
        for t in range(max_steps):
            print("current state", state)
            action = agent.select_action(np.array(state))
            print("current action", action)
            state, reward, done, _ = env.step(action)
            env.render()
            # time.sleep(0.1)
            eps_reward += reward
            # u_list[t] = action
        all_rewards[i] = eps_reward

    print(f"mean episodic reward over 200 time steps (rf_num = {rf_num}, learn_rf = {learn_rf}): ",
          np.mean(all_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=0, type=int)
    parser.add_argument("--alg", default="sac")  # Alg name (sac, vlsac,rfsac)
    parser.add_argument("--env", default="quadrotor")  # Environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e5, type=float)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=1024, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    parser.add_argument("--sigma", default=0., type=float)  # noise for noisy environment
    parser.add_argument("--rand_feat_num", default=1024, type=int)
    parser.add_argument("--learn_rf", default="False")  # string indicating if learn_rf is false or no
    parser.add_argument("--dir_name", default="2023-04-12-00-40-34")
    args = parser.parse_args()

    sigma = args.sigma

    if args.env == "Pendulum-v1":
        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        max_length = env._max_episode_steps
        env = noisyPendulumEnv(sigma=sigma)
        # log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}/T={args.max_timesteps}/rf_num={args.rand_feat_num}/learn_rf={learn_rf}'
    elif args.env == 'quadrotor':
        CONFIG_FACTORY = ConfigFactory()
        CONFIG_FACTORY.parser.set_defaults(overrides=['./our_env/env_configs/stabilization.yaml'])
        config = CONFIG_FACTORY.merge()

        CONFIG_FACTORY_EVAL = ConfigFactory()
        CONFIG_FACTORY_EVAL.parser.set_defaults(overrides=['./our_env/env_configs/stabilization_eval.yaml'])
        config_eval = CONFIG_FACTORY_EVAL.merge()

        args.fixed_steps = int(config.quadrotor_config['episode_len_sec'] * config.quadrotor_config['ctrl_freq'])
        args.config = deepcopy(config)
        args.config_eval = deepcopy(config_eval)
        # config.quadrotor_config['gui'] = False
        # args.config_eval.quadrotor_config['gui'] = False
        # env = make(args.env, **config.quadrotor_config)
        env = make(args.env, **config_eval.quadrotor_config)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        max_length = 360
    print("hey", args.learn_rf)
    learn_rf = True if args.learn_rf == "True" else False
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "sigma": sigma,
        "rand_feat_num": args.rand_feat_num,
        "learn_rf": learn_rf
    }

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
        log_path = f'/home/mht/PycharmProjects/lvrep-rl-cloned/exp/{args.env}/{args.alg}/{args.dir}/{args.seed}/T={args.max_timesteps}/' + args.dir_name
        agent.actor.load_state_dict(torch.load(log_path + "/actor.pth"))
        agent.critic.load_state_dict(torch.load(log_path + "/critic.pth"))
    elif args.alg == 'vlsac':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = vlsac_agent.VLSACAgent(**kwargs)
    elif args.alg == 'rfsac':
        agent = rfsac_agent.RFSACAgent(**kwargs)
        log_path = f'/home/mht/PycharmProjects/lvrep-rl-cloned/exp/{args.env}/{args.alg}/{args.dir}/{args.seed}/T={args.max_timesteps}/rf_num={args.rand_feat_num}/learn_rf={learn_rf}/sigma=0.0/' + args.dir_name
        actor = DiagGaussianActor(obs_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim, hidden_depth=2,
                                  log_std_bounds=[-5., 2.]).to(agent.device)
        critic = RFVCritic(s_dim=state_dim, sigma=sigma, rand_feat_num=args.rand_feat_num, learn_rf=learn_rf).to(agent.device)
        actor.load_state_dict(torch.load(log_path + "/actor.pth"))
        critic.load_state_dict(torch.load(log_path + "/critic.pth"))
        agent.actor = actor
        agent.critic = critic
    max_length = 1000

    main(env, log_path, agent, rf_num=args.rand_feat_num, learn_rf=learn_rf, state_dim=state_dim)
