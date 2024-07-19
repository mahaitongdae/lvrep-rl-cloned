import sys
import os
from gymnasium.envs.registration import register
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)
from repr_control.envs.custom_env import CustomEnv
from repr_control.envs.articulate_fh import ArticulateParking

register('ArticulateFiniteHorizon-v0',
         entry_point='repr_control.envs:ArticulateParking',
         max_episode_steps=500)