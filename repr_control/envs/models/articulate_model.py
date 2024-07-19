import torch
import numpy as np

def dynamics(state, action):
    l = 4.9276  # tractor length
    R = 8.5349  # turning radius
    d1 = 15.8496  # trailer length
    delta_max = np.pi / 6
    dt = 0.05

    x, y, th0, dth, v, delta, t = state
    acc = action[:, 0]
    delta_rate = action[:, 1] * delta_max / 4
    normalized_steer = torch.tan(delta) * R / l

    ds = torch.vstack([
        v * torch.cos(th0),
        v * torch.sin(th0),
        v * normalized_steer / R,
        -1 * v * (d1 * normalized_steer + torch.sin(dth) * R) / (R * d1),
        acc,
        delta_rate,
        1.0,
    ]).T

    stp1 = state + ds * dt 
    return stp1

def reward(state, action, t):
    x, y, th0, dth, v, delta, t = state
    acc, delta_rate = action
    if t < 249:
        reward = -1e-4 * (x ** 2 + y ** 2 + 10 * th0 ** 2 + 10 * dth ** 2 + acc ** 2 + delta_rate ** 2)
    else:
        reward = -1 * (10 * x ** 2 + 10 * y ** 2 + 500 * th0 ** 2 + 500 * dth ** 2)
    return reward

def initial_distribution(batch_size):

    high = np.array(
            [
                10,
                3,
                np.pi / 6,
                np.pi / 6,
                0.0,
                0.0,
                0.0
            ],
            dtype=np.float32,
        )

    reset_std = np.array(
        [3.0,
            0.5,
            np.pi / 12,
            np.pi / 12,
            0.0,
            0.0,
            0.0
            ]
    )
    # self.state = self.np_random.uniform(low=-1 * high, high=high)
    state = np.random.normal(np.zeros_like(reset_std), reset_std)
    state = np.clip(state, -high, high)
    return state


