import numpy as np
import torch
import gym
import argparse
import os
from copy import deepcopy
import time

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent

# from our_env.noisy_pend import noisyPendulumEnv
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


if __name__ == '__main__':
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./our_env/env_configs/stabilization.yaml'])
    config = CONFIG_FACTORY.merge()

    CONFIG_FACTORY_EVAL = ConfigFactory()
    CONFIG_FACTORY_EVAL.parser.set_defaults(overrides=['./our_env/env_configs/stabilization.yaml'])
    config_eval = CONFIG_FACTORY_EVAL.merge()

    fixed_steps = int(config.quadrotor_config['episode_len_sec'] * config.quadrotor_config['ctrl_freq'])
    config = deepcopy(config)
    config_eval = deepcopy(config_eval)
    # config.quadrotor_config['gui'] = False
    # args.config_eval.quadrotor_config['gui'] = False
    env = make('quadrotor', **config.quadrotor_config)

    env.reset()

    for i in range(10):
        action = np.array([0.1, 0.1])
        obs,rew,done, info = env.step(action)
        print(rew)
        env.render()
        time.sleep(0.1)


