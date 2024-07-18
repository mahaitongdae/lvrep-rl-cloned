from typing import Optional, Union, Tuple, Callable

import numpy as np
import torch
import gym
from gym import spaces
from gym.core import ObsType, ActType
from gym.error import DependencyNotInstalled


class CustomEnv(gym.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, dynamics: Callable, rewards: Callable, initial_distribution: Callable,
                    state_range: list, action_range: list, sigma: float):
        self.observation_space = spaces.Box(low=np.array(state_range[0]), high=np.array(state_range[1]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(action_range[0]), high=np.array(action_range[1]), dtype=np.float32)
        self.dynamics = dynamics
        self.rewards = rewards
        self.initial_distribution = initial_distribution
        self.sigma = sigma

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.initial_distribution(batch_size = 1).squeeze().numpy()

        return self.state

    def step(
        self, action: ActType
    ) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        state = torch.from_numpy(self.state[np.newaxis, ...]).float()
        if isinstance(action, float):
            action = np.array([action])
        action = torch.from_numpy(action[np.newaxis, ...]).float()
        with torch.no_grad():
            next_state = self.dynamics(state, action)
        true_next_state = next_state.squeeze().numpy()
        noisy_next_state = true_next_state + np.random.normal(0, self.sigma, true_next_state.shape)
        self.state = np.clip(noisy_next_state, self.observation_space.low, self.observation_space.high)
        reward = self.rewards(state, action)
        reward = reward.squeeze().item()
        done = False
        info = {}
        return self.state, reward, done, info

def test_custom_env():
    from define_problem import dynamics, rewards, initial_distribution, state_range, action_range, sigma
    env = CustomEnv(dynamics, rewards, initial_distribution, state_range, action_range, sigma)

    print(env.reset())
    for i in range(10):
        print(env.step(env.action_space.sample()))

if __name__ == '__main__':
    test_custom_env()