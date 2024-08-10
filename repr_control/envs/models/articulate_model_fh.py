import torch
import numpy as np
state_dim = 7                       # state dimension
action_dim = 2                      # action dimension
state_range = [[
                -20,
                -10,
                -np.pi,
                -np.pi / 2,
                -0.6,
                -np.pi / 6,
            ],
    [
        20,
        10,
        np.pi,
        np.pi / 2,
        0.6,
        np.pi / 6,
    ],
]           # low and high. We set bound on the state to ensure stable training.
action_range = [[-1, -1], [1, 1]]          # low and high
max_step = 250                      # maximum rollout steps per episode
sigma = 0.0                          # noise standard deviation.
env_name = 'Parking'
assert len(action_range[0]) == len(action_range[1]) == action_dim
def dynamics(state, action):
    l = 4.9276  # tractor length
    R = 8.5349  # turning radius
    d1 = 15.8496  # trailer length
    delta_max = np.pi / 6
    dt = 0.05

    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc = action[:, 0] * 2
    delta_rate = action[:, 1] * delta_max / 2
    normalized_steer = torch.tan(delta) * R / l

    ds = torch.vstack([
        v * torch.cos(th0),
        v * torch.sin(th0),
        v * normalized_steer / R,
        -1 * v * (d1 * normalized_steer + torch.sin(dth) * R) / (R * d1),
        acc,
        delta_rate,
    ]).T

    stp1 = state + ds * dt
    stp1[:, -2].clip_(-2.0, 2.0)
    stp1[:, -1].clip_(-delta_max, delta_max)
    return stp1

def rewards(state, action, terminal = False):
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc, delta_rate = torch.unbind(action, dim=1)
    if not terminal:
        reward = -1e-4 * (x ** 2 + y ** 2
                          + 10 * th0 ** 2
                          + 10 * dth ** 2
                          + v ** 2
                          + delta ** 2
                          + 10 * acc ** 2
                          + 10 * delta_rate ** 2)
    else:
        reward = -1 * (10 * x ** 2 + 10 * y ** 2 + 100 * th0 ** 2 + 100 * dth ** 2)
    return reward

def initial_distribution(batch_size):

    # high = np.array(
    #         [
    #             10,
    #             3,
    #             np.pi / 6,
    #             np.pi / 6,
    #             0.0,
    #             0.0,
    #             0.0
    #         ],
    #         dtype=np.float32,
    #     )
    #
    # reset_std = np.array(
    #     [3.0,
    #         0.5,
    #         np.pi / 12,
    #         np.pi / 12,
    #         0.0,
    #         0.0,
    #         0.0
    #         ]
    # )
    # self.state = self.np_random.uniform(low=-1 * high, high=high)
    state = np.random.uniform(low=np.array([2.0, 0.0, - np.pi / 12, 0.0, 0.0, 0.0]),
                              high=np.array([5.0, 1.5, np.pi / 12, 0.0 ,0.0, 0.0]),
                              size=(batch_size, 6))
    state[:, 3] = - state[:, 2] # we sample theta_1 and calculate theta_1 - theta_0
    return torch.from_numpy(state)


