"""
We need to define the nonlinear control problems in this file.
"""

import torch
import numpy as np

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 2  # state dimension
action_dim = 1  # action dimension
state_range = [[-np.inf, -np.inf], [np.inf, np.inf]]  # low and high. We set bound on the state to ensure stable training.
action_range = [[-np.inf], [np.inf]]  # low and high
max_step = 200  # maximum rollout steps per episode
sigma = 0.00  # noise standard deviation.
env_name = 'linear'
assert len(action_range [0]) == len(action_range [1]) == action_dim
device = torch.device('cuda')


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
    A = torch.tensor([[0.9, 0.2],
                      [0.0, 0.8]], device=device)
    assert state.shape[1] == 2
    bmatmul = torch.vmap(torch.matmul, in_dims=(None, 0))
    stp1 = bmatmul(A, state)
    return stp1


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
    return torch.ones((state.shape[0]))


def initial_distribution(batch_size: int) -> torch.Tensor:
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros(2, device=device),
                                                                   covariance_matrix=torch.tensor([[1.0,0.5],
                                                                                                   [0.5, 1.0]], device=device))
    return dist.sample([batch_size])
