"""
We need to define the nonlinear control problems in this file.
"""

import torch
import numpy as np

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 5  # state dimension
action_dim = 1  # action dimension
state_range = [[-4.8, -10, -0.418, ],
               [1, 1, 8]]  # low and high. We set bound on the state to ensure stable training.
action_range = [[-10], [10]]  # low and high
max_step = 200  # maximum rollout steps per episode
sigma = 0.05  # noise standard deviation.
env_name = 'CartPole'
assert len(action_range[0]) == len(action_range[1]) == action_dim


########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The dynamics. Needs to be written in pytorch to enable auto differentiation.
    The input and outputs should be 2D Tensors, where the first dimension should be batch size, and the second dimension
    is the state. For example, the pendulum state will looks like
    [[cos(theta), sin(theta), dot theta],
     [cos(theta), sin(theta), dot theta],
     ...,
     [cos(theta), sin(theta), dot theta]
     ]

    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim]
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    next_state       torch.Tensor, [batch_size, state_dim]
    -------

    """
    masscart = 1.0
    masspole = 0.1
    length = 0.5
    total_mass = masspole + masscart
    polemass_length = masspole * length
    dt = 0.02
    gravity = 9.81
    new_states = torch.empty_like(state).to(device=state.device)
    new_states[:, 0] = state[:, 0] + dt * state[:, 1]
    costheta = state[:, -3]
    sintheta = state[:, -2]
    theta_dot = state[:, -1]
    theta = torch.atan2(sintheta, costheta)
    new_theta = theta + dt * theta_dot
    new_states[:, -3] = torch.cos(new_theta)
    new_states[:, -2] = torch.sin(new_theta)
    # new_states[:, 2] = states[:, 2] + dt * states[:, 3]
    # theta = states[:, 2]

    force = torch.squeeze(10. * action)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = 1. / total_mass * (
            force + polemass_length * theta_dot ** 2 * sintheta
    )
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    new_states[:, 1] = state[:, 1] + dt * xacc
    new_states[:, 4] = theta_dot + dt * thetaacc
    return new_states


def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The reward. Needs to be written in pytorch to enable auto differentiation.

    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim]
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    rewards       torch.Tensor, [batch_size,]
    -------

    """
    th = torch.atan2(state[:, -2], state[:, -3])  # torch.unsqueeze(, dim=1)  # -2 is sin, -3 is cos
    # th = torch.where(obs[:, -3] >= 0., th, th + torch.pi)
    ## arctan only return [-pi/2, pi/2].
    reward = - ((torch.remainder(th, 2 * torch.pi) - torch.pi) ** 2)
    return reward


def initial_distribution(batch_size: int) -> torch.Tensor:
    th = 2 * np.pi * torch.rand((batch_size)) - np.pi
    thdot = 2 * torch.rand((batch_size)) - 1
    return torch.vstack([torch.cos(th),
                         torch.sin(th),
                         thdot]).T
