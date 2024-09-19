from __future__ import annotations

from typing import Optional, Union, Tuple, Callable, SupportsFloat, Any

import numpy as np
import torch
import gymnasium
from gymnasium import spaces
from gymnasium.core import ObsType, ActType


class CustomEnv(gymnasium.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, dynamics: Callable, rewards: Callable, initial_distribution: Callable,
                    state_range: list, action_range: list, sigma: float):
        self.observation_space = spaces.Box(low=np.array(state_range[0], dtype=np.float32),
                                            high=np.array(state_range[1], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(action_range[0], dtype=np.float32),
                                       high=np.array(action_range[1], dtype=np.float32), dtype=np.float32)
        self.dynamics = dynamics
        self.rewards = rewards
        self.initial_distribution = initial_distribution
        self.sigma = sigma

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.initial_distribution(batch_size = 1).squeeze().float().numpy()

        return self.state, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        state = torch.from_numpy(self.state[np.newaxis, ...]).float()
        if isinstance(action, float):
            action = np.array([action])
        action = torch.from_numpy(action[np.newaxis, ...]).float()
        with torch.no_grad():
            next_state = self.dynamics(state, action)
        true_next_state = next_state.squeeze().numpy()
        noisy_next_state = true_next_state + np.random.normal(0, self.sigma, true_next_state.shape)
        self.state = np.clip(noisy_next_state, self.observation_space.low, self.observation_space.high, dtype=np.float32)
        reward = self.rewards(state, action)
        reward = reward.squeeze().item()
        done = False
        info = {}
        return self.state, reward, done, False, info