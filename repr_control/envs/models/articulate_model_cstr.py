import torch
torch.set_default_dtype(torch.float32)
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
L_TRACTOR = 4.9276  # tractor length
R = 8.5349  # turning radius
L_TRAILER = 15.8496  # trailer length
W = 3.1162
assert len(action_range[0]) == len(action_range[1]) == action_dim
def dynamics(state, action, trailer_length):
    delta_max = np.pi / 6
    dt = 0.05
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc = action[:, 0] * 2
    delta_rate = action[:, 1] * delta_max / 2
    normalized_steer = torch.tan(delta) * R / L_TRACTOR

    ds = torch.vstack([
        v * torch.cos(th0),
        v * torch.sin(th0),
        v * normalized_steer / R,
        -1 * v * (trailer_length * normalized_steer + torch.sin(dth) * R) / (R * trailer_length),
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

def xy_rewards(state,action):
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc, delta_rate = torch.unbind(action, dim=1)
    reward = -1e-3 * (x ** 2 + y ** 2
                      +  acc ** 2
                      +  delta_rate ** 2)
    return reward

def one_hot_rewards(state, action, xf):
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
    constraints = terminal_constraints(state, xf)
    max_constraints = torch.max(constraints, dim=1)[0]
    penalty = torch.where(max_constraints > torch.zeros_like(max_constraints),
                          -1 * torch.ones_like(max_constraints),
                          torch.zeros_like(max_constraints))
    return rewards + penalty

def terminal_constraints(state, xf=None):
    if xf is None:
        xf = torch.zeros_like(state).to(state.device)
    error = torch.abs(xf - state)
    x, y, th0, dth, v, delta = torch.unbind(error, dim=1)
    constraints = torch.vstack([
        x - 0.05,
        y - 0.05,
        th0 - 1 * torch.pi / 180,
        dth - 1 * torch.pi / 180,
    ]).T
    return constraints

def true_terminal_constraints(state, xf=None):
    if xf is None:
        xf = torch.zeros_like(state).to(state.device)
    error = torch.abs(xf - state)
    x, y, th0, dth, v, delta = torch.unbind(error, dim=1)
    constraints = torch.vstack([
        x - 0.1,
        y - 0.1,
        th0 - 2 * torch.pi / 180,
        dth - 2 * torch.pi / 180,
    ]).T
    return constraints

def initial_distribution(batch_size):

    state = np.random.uniform(low=np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                  high=np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                  size=(batch_size, 6))
    state = torch.from_numpy(state)# .float()
    goal = goal_distribution(batch_size)
    init_from_goal = torch.vmap(transform_from_intial_to_goal_full_state)(goal[:, :3], state.unsqueeze(1))
    init_from_goal = init_from_goal.squeeze(1)
    obstacle = obstacle_distribution_goal_frame(batch_size, goal)
    trailer_length = np.random.uniform(low=np.array([12.0]),
                                        high=np.array([16.0]),
                                        size=(batch_size, 1))
    trailer_length = torch.from_numpy(trailer_length)# .float()
    return torch.hstack([state, init_from_goal, obstacle, trailer_length])

def goal_distribution(batch_size):

    goal = np.random.uniform(low=np.array([20.0, 17.11, np.pi / 2, 0.0, 0.0, 0.0]),
                              high=np.array([20.0, 17.11, np.pi / 2, 0.0, 0.0, 0.0]),
                              size=(batch_size, 6))
    return torch.from_numpy(goal)

def obstacle_distribution(batch_size):
    """

    Parameters
    ----------
    batch_size: int,

    Returns
    -------
    batch_flatten_obs: torch.Tensor [bs, 32]

    """
    obstacle = np.array([
        [-20., 5.],
        [-20., 35.],
        [ 15.,  35.],
        [15., 5.],
        [-20., -40.],
        [-20., -5.],
        [15., -5.],
        [15., -40.],
        [35., -40.],
        [35., 10.],
        [80., 10.],
        [ 80., -40.],
        [35., 30.],
        [35., 80.],
        [80., 80.],
        [80., 30.], ])
    flatten_obs = np.reshape(obstacle, [1, -1])
    batch_flatten_obs = torch.from_numpy(np.repeat(flatten_obs, batch_size, axis=0))
    return batch_flatten_obs

def obstacle_distribution_goal_frame(batch_size, goal_xyt):

    obstacles = obstacle_distribution(batch_size)
    assert obstacles.shape[0] == goal_xyt.shape[0]
    obstacles = obstacles.reshape((batch_size, -1, 2))
    obstacles_goal_frame = torch.vmap(transform_from_intial_to_goal)(goal_xyt, obstacles)
    return obstacles_goal_frame.reshape((batch_size, -1))


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

def transform_from_intial_to_goal(goal_xyt: torch.Tensor, points_initial: torch.Tensor):
    """

    Parameters
    ----------
    goal_xyt: torch.Tensor [3], x, y, theta
    points_initial: torch.Tensor [N, 2], x, y

    Returns
    -------
    points_goal: torch.Tensor [N, 2], x, y in goal frame

    """
    # goalx, goaly, goal_theta = goal_xyt
    goal_theta = goal_xyt[[2]]
    goal_xy = goal_xyt[:2]
    # rotmat_goal_to_init = np.array([[np.cos(goal_theta), -np.sin(goal_theta)],
    #                         [np.sin(goal_theta), np.cos(goal_theta)]])
    # rotmat_init_to_goal = torch.empty([2, 2])
    # rotmat_init_to_goal[0, 0] = torch.cos(goal_theta)
    # rotmat_init_to_goal[0, 1] = torch.sin(goal_theta)
    # rotmat_init_to_goal[1, 0] = - torch.sin(goal_theta)
    # rotmat_init_to_goal[1, 1] = torch.cos(goal_theta)
    rotmat_init_to_goal = torch.cat((torch.cos(goal_theta),
                        torch.sin(goal_theta),
                        - torch.sin(goal_theta),
                        torch.cos(goal_theta))).reshape((2, 2))
    # rotmat_init_to_goal = torch.tensor([[torch.cos(goal_theta), torch.sin(goal_theta)],
    #                         [-torch.sin(goal_theta), torch.cos(goal_theta)]])
    translation_init_to_goal = rotmat_init_to_goal @ goal_xy.reshape([2, 1]) # 2 * 1
    points_goal = rotmat_init_to_goal @ points_initial.T - translation_init_to_goal # 2 * N
    points_goal = points_goal.T
    return points_goal

def transform_from_intial_to_goal_full_state(goal_xyt: torch.Tensor,
                                             states_initial: torch.Tensor):
    """

    Parameters
    ----------
    goal_xyt: torch.Tensor [3], x, y, theta
    states_initial: torch.Tensor [N, 6], x, y, theta, dtheta, v, delta

    Returns
    -------

    """
    # goalx, goaly, goal_theta = goal_state
    # rotmat_goal_to_init = np.array([[np.cos(goal_theta), -np.sin(goal_theta)],
    #                         [np.sin(goal_theta), np.cos(goal_theta)]])
    points_initial = states_initial[:, :2]
    points_goal = transform_from_intial_to_goal(goal_xyt, points_initial)
    theta_goal = states_initial[:, [2]] - goal_xyt[2]
    other_states = states_initial[:, 3:]

    return torch.cat([points_goal, theta_goal, other_states], dim=-1)

def plot_obstacles(frame = 'goal'):
    from matplotlib import pyplot as plt
    # obstacle = obstacle_distribution(1).squeeze().reshape([-1, 2])
    # goal = goal_distribution(1).squeeze()[:3]
    # if frame == 'goal':
    #     obstacle = transform_from_intial_to_goal(goal, obstacle)
    goal = goal_distribution(1)
    obstacle = obstacle_distribution_goal_frame(1, goal[:, :3]).squeeze().reshape((-1, 2))

    obstacles = torch.split(obstacle, 4, dim=0)
    print(obstacles)
    plt.figure(figsize=(8, 8))
    for obs in obstacles:
        plt.scatter(obs[:, 0], obs[:, 1])
    plt.axis('equal')
    plt.show()

def test_transform_full_state():
    goal_from_init = goal_distribution(1)
    goal_from_goal = torch.vmap(transform_from_intial_to_goal_full_state)(goal_from_init[:, :3], goal_from_init.unsqueeze(1))
    print(goal_from_goal)
    init_from_init = torch.zeros((1, 6))
    init_from_goal = torch.vmap(transform_from_intial_to_goal_full_state)(goal_from_init[:, :3], init_from_init.unsqueeze(1))
    print(init_from_goal)


if __name__ == '__main__':
    # print(initial_distribution(256).shape)
    print(initial_distribution(1))

    # goal = goal_distribution(1)
    # obstacle_distribution_goal_frame(1, goal[:, :3])


