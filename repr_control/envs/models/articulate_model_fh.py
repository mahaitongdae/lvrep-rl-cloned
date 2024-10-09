import torch
import numpy as np
state_dim = 6                       # state dimension
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
        reward = -1 * (1 * x ** 2 + 10 * y ** 2 + 100 * th0 ** 2 + 100 * (th0 + dth) ** 2)
    return reward

def one_hot_rewards(state, action):
    # x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc, delta_rate = torch.unbind(action, dim=1)
    # if not terminal:
    #     reward = -1e-4 * (x ** 2 + y ** 2
    #                       + 10 * th0 ** 2
    #                       + 10 * dth ** 2
    #                       + v ** 2
    #                       + delta ** 2
    #                       + 10 * acc ** 2
    #                       + 10 * delta_rate ** 2)
    # else:
    #     reward = -1 * (1 * x ** 2 + 10 * y ** 2 + 100 * th0 ** 2 + 100 * (th0 + dth) ** 2)
    # return reward
    rewards = -1 * acc ** 2 - 1 * delta_rate ** 2
    constraints = terminal_constraints(state)
    max_constraints = torch.max(constraints, dim=1)[0]
    penalty = torch.where(max_constraints > torch.zeros_like(max_constraints), -1 * torch.ones_like(max_constraints), torch.zeros_like(max_constraints))
    return rewards + penalty

def terminal_constraints(state):
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    constraints = torch.vstack([
        torch.abs(x) - 0.1,
        torch.abs(y) - 0.1,
        torch.abs(th0) - 2 * torch.pi / 180,
        torch.abs(dth) - 2 * torch.pi / 180,
    ]).T
    return constraints


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
    state = np.random.uniform(low=np.array([2.0, 0.5, - np.pi / 12, 0.0, 0.0, 0.0]),
                              high=np.array([5.0, 1.5, np.pi / 12, 0.0 ,0.0, 0.0]),
                              size=(batch_size, 6))
    state[:, 3] = - state[:, 2] # we sample theta_1 and calculate theta_1 - theta_0
    return torch.from_numpy(state)

def evaluate_initial_states(grid_size):

    x = np.linspace(2., 5, grid_size)
    y = np.linspace(0.5, 1.5, grid_size)
    th0 = np.linspace(-np.pi / 12, np.pi / 12, grid_size)
    # Create the grid
    X, Y, TH0 = np.meshgrid(x, y, th0, indexing='ij')

    grid_x = X.ravel()
    grid_y = Y.ravel()
    grid_th0 = TH0.ravel()
    grid_dth = -1 * grid_th0
    grid_v = np.zeros_like(grid_x)
    grid_delta = np.zeros_like(grid_x)

    init_states = np.vstack([grid_x, grid_y, grid_th0, grid_dth, grid_v, grid_delta]).T

    return torch.from_numpy(init_states)


