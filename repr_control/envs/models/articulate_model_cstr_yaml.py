import torch
torch.set_default_dtype(torch.float32)
import numpy as np
import yaml
import repr_control
pkg_dir = repr_control.__path__[0]

MAP_ID = '2'
TASK = 'forward_left'

def load_config_from_map(task=TASK, map_id=MAP_ID):
    if isinstance(map_id, int):
        map_id = int(map_id)
    
    # load setup from yaml file
    with open(f"{pkg_dir}/config/map{map_id}.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    task_config = config['tasks'][task]
    obstacles = config['obstacles']
    trailer_config = config['trailer_length']
    
    return task_config, obstacles, trailer_config, config

# load default config

task_config, obstacles, trailer_config, _ = load_config_from_map()

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
    dt = 0.1
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

def xy_rewards(state,action, xf=None):
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc, delta_rate = torch.unbind(action, dim=1)
    reward = -1e-3 * (x ** 2 + y ** 2
                      +  acc ** 2
                      +  delta_rate ** 2)
    return reward

def zero_rewards(state, action):
    return torch.zeros_like(state[:, 0])

def one_hot_rewards(state, action, xf=None):
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
                          torch.zeros_like(max_constraints),
                          torch.ones_like(max_constraints))
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



def initial_distribution(batch_size, 
                         task_config=task_config, 
                         obstacles=obstacles, 
                         trailer_config=trailer_config):
    """
    Parameters
    ----------
    batch_size: int,
    
    return state from goal coord, initial states from goal coord, obstacle in goal coord, and trailer length
    
    Returns
    -------
    init_state: torch.Tensor [bs, 6], x, y, theta, dtheta, v, delta
    """
    
    

    init_state_world = initial_state_distribution(batch_size, task_config)
    # init_state_world = torch.from_numpy(init_state_world)# .float()
    goal_world = goal_state_distribution(batch_size, task_config)
    init_from_goal = torch.vmap(transform_to_goal_full_state)(goal_world[:, :3], init_state_world.unsqueeze(1)) 
    # The function can transfer N points and we actually transfer one, so it is [bs, 1, 6]
    init_from_goal = init_from_goal.squeeze(1)
    obstacles_goal_frame = obstacle_distribution_goal_frame(batch_size, obstacles, goal_world[:, :3])
    trailer_length = np.random.uniform(low=np.array([trailer_config['min']]),
                                        high=np.array([trailer_config['max']]),
                                        size=(batch_size, 1))
    trailer_length = torch.from_numpy(trailer_length)# .float()
    return torch.hstack([init_from_goal, init_from_goal, obstacles_goal_frame, trailer_length])

def initial_state_distribution(batch_size: int, config: dict):
    """
    Parameters
    ----------
    batch_size: int,
    
    Returns
    -------
    init_state: torch.Tensor [bs, 6], x, y, theta, dtheta, v, delta
    """
    # forward
    state_low = config['init']['min'] + [ 0.0, 0.0, 0.0]
    state_high = config['init']['max'] + [ 0.0, 0.0, 0.0]
    state = np.random.uniform(low=np.array(state_low),
                                  high=np.array(state_high),
                                  size=(batch_size, 6))
    # reverse
    # state = np.random.uniform(low=np.array([ 0.0, 35.0, np.pi / 2, 0.0, 0.0, 0.0]),
    #                               high=np.array([  0.0, 25.0, np.pi / 2, 0.0, 0.0, 0.0]),
    #                               size=(batch_size, 6))
    state = torch.from_numpy(state)# .float()
    return state

def goal_state_distribution(batch_size: int, config: dict):
    """
    Parameters
    ----------
    batch_size: int,
    
    Returns
    -------
    goal: torch.Tensor [bs, 6], x, y, theta, dtheta, v, delta
    """
    state_low = config['goal']['min'] + [ 0.0, 0.0, 0.0]
    state_high = config['goal']['max'] + [ 0.0, 0.0, 0.0]

    goal = np.random.uniform(low=np.array(state_low),
                              high=np.array(state_high),
                              size=(batch_size, 6))
    # right turn
    # goal = np.random.uniform(low=np.array([24.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    #                           high=np.array([42.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    #                           size=(batch_size, 6))
    # reverse left
    # goal = np.random.uniform(low=np.array([8.0, 5.0, np.pi, 0.0, 0.0, 0.0]),
    #                           high=np.array([12.0, 5.0, np.pi, 0.0, 0.0, 0.0]),
    #                           size=(batch_size, 6))
    # reverse right
    # goal = np.random.uniform(low=np.array([-12.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    #                           high=np.array([-12.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    #                           size=(batch_size, 6))
    return torch.from_numpy(goal)

def obstacle_distribution(batch_size, obstacle, ):
    """
    Obstacle distribution in the world frame.
    
    Parameters
    ----------
    batch_size: int,

    Returns
    -------
    batch_flatten_obs: torch.Tensor [bs, 32]

    """
    # obstacle = np.array([
    #     [-20., 5.],
    #     [-20., 35.],
    #     [ 15.,  35.],
    #     [15., 5.],
    #     [-20., -40.],
    #     [-20., -5.],
    #     [15., -5.],
    #     [15., -40.],
    #     [35., -40.],
    #     [35., 10.],
    #     [80., 10.],
    #     [ 80., -40.],
    #     [35., 30.],
    #     [35., 80.],
    #     [80., 80.],
    #     [80., 30.], ])
    # obstacle = config['obstacles']
    obstacle = np.array([
        obstacle
    ])
    flatten_obs = np.reshape(obstacle, [1, -1])
    batch_flatten_obs = torch.from_numpy(np.repeat(flatten_obs, batch_size, axis=0))
    return batch_flatten_obs

def obstacle_distribution_goal_frame(batch_size: int, config, goal_xyt: torch.Tensor):
    """
    Get obstacle distribution in goal frame.
    Parameters
    ----------
    batch_size: int,
    goal_xyt: torch.Tensor [bs, 3], x, y, theta
    
    Returns
    -------
    obstacles_goal_frame: torch.Tensor [bs, 32]
    """

    obstacles = obstacle_distribution(batch_size, config)
    assert obstacles.shape[0] == goal_xyt.shape[0]
    obstacles = obstacles.reshape((batch_size, -1, 2))
    obstacles_goal_frame = torch.vmap(transform_to_goal)(goal_xyt, obstacles)
    return obstacles_goal_frame.reshape((batch_size, -1))


# def evaluate_initial_states(grid_size):

#     x = np.linspace(2., 5, grid_size)
#     y = np.linspace(0.5, 1.5, grid_size)
#     th0 = np.linspace(-np.pi / 12, np.pi / 12, grid_size)
#     # Create the grid
#     X, Y, TH0 = np.meshgrid(x, y, th0, indexing='ij')

#     grid_x = X.ravel()
#     grid_y = Y.ravel()
#     grid_th0 = TH0.ravel()
#     grid_dth = -1 * grid_th0
#     grid_v = np.zeros_like(grid_x)
#     grid_delta = np.zeros_like(grid_x)

#     init_states = np.vstack([grid_x, grid_y, grid_th0, grid_dth, grid_v, grid_delta]).T

#     return torch.from_numpy(init_states)

def evaluate_initial_states(bs):
    return initial_distribution(bs)

def transform_to_goal(goal_xyt: torch.Tensor, points: torch.Tensor):
    """

    Parameters
    ----------
    goal_xyt: torch.Tensor [3], x, y, theta
    points_initial: torch.Tensor [N, 2], x, y, in world coordinate

    Returns
    -------
    points_goal: torch.Tensor [N, 2], x, y in goal frame

    """
    # goalx, goaly, goal_theta = goal_xyt
    goal_theta = goal_xyt[[2]]
    goal_xy = goal_xyt[:2].float()
    # rotmat_goal_to_init = np.array([[np.cos(goal_theta), -np.sin(goal_theta)],
    #                         [np.sin(goal_theta), np.cos(goal_theta)]])
    # rotmat_init_to_goal = torch.empty([2, 2])
    # rotmat_init_to_goal[0, 0] = torch.cos(goal_theta)
    # rotmat_init_to_goal[0, 1] = torch.sin(goal_theta)
    # rotmat_init_to_goal[1, 0] = - torch.sin(goal_theta)
    # rotmat_init_to_goal[1, 1] = torch.cos(goal_theta)
    rotmat_world_from_goal = torch.cat((torch.cos(goal_theta),
                        torch.sin(goal_theta),
                        - torch.sin(goal_theta),
                        torch.cos(goal_theta))).reshape((2, 2)).float()
    # rotmat_init_to_goal = torch.tensor([[torch.cos(goal_theta), torch.sin(goal_theta)],
    #                         [-torch.sin(goal_theta), torch.cos(goal_theta)]])
    translation_init_to_goal = rotmat_world_from_goal @ goal_xy.reshape([2, 1]) # 2 * 1
    points_goal = rotmat_world_from_goal @ points.float().T - rotmat_world_from_goal @ goal_xy.reshape([2, 1]) # 2 * N
    points_goal = points_goal.T
    return points_goal

def transform_to_goal_full_state(goal_xyt: torch.Tensor,
                                states_world: torch.Tensor):
    """

    Parameters
    ----------
    goal_xyt: torch.Tensor [3], x, y, theta
    states_world: torch.Tensor [N, 6], x, y, theta, dtheta, v, delta

    Returns
    -------

    """
    # goalx, goaly, goal_theta = goal_state
    # rotmat_goal_to_init = np.array([[np.cos(goal_theta), -np.sin(goal_theta)],
    #                         [np.sin(goal_theta), np.cos(goal_theta)]])
    points_initial = states_world[:, :2]
    points_goal = transform_to_goal(goal_xyt, points_initial)
    theta_goal = states_world[:, [2]] - goal_xyt[2]
    other_states = states_world[:, 3:]

    return torch.cat([points_goal, theta_goal, other_states], dim=-1)

def plot_obstacles(frame = 'goal'):
    from matplotlib import pyplot as plt
    # obstacle = obstacle_distribution(1).squeeze().reshape([-1, 2])
    # goal = goal_distribution(1).squeeze()[:3]
    # if frame == 'goal':
    #     obstacle = transform_from_intial_to_goal(goal, obstacle)
    # load setup from yaml file
    with open(f"{pkg_dir}/config/map{MAP_ID}.yaml", "r") as file:
        config = yaml.safe_load(file)
    goal = goal_state_distribution(1)
    obstacle = obstacle_distribution_goal_frame(1, config, goal[:, :3]).squeeze().reshape((-1, 2))

    obstacles = torch.split(obstacle, 4, dim=0)
    print(obstacles)
    plt.figure(figsize=(8, 8))
    for obs in obstacles:
        plt.scatter(obs[:, 0], obs[:, 1])
    plt.axis('equal')
    plt.savefig('obstacles.png')

def test_transform_full_state():
    goal_world = goal_state_distribution(1)
    print("goal state:", goal_world)
    goal_from_goal = torch.vmap(transform_to_goal_full_state)(goal_world[:, :3], goal_world.unsqueeze(1))
    print(goal_from_goal)
    init_world = initial_state_distribution(1)
    print("init state:", init_world)
    init_from_goal = torch.vmap(transform_to_goal_full_state)(goal_world[:, :3], init_world.unsqueeze(1))
    print(init_from_goal)


if __name__ == '__main__':
    print(initial_distribution(256).shape)
    print(initial_distribution(1))
    # test_transform_full_state()

    # goal = goal_distribution(1)
    # obstacle_distribution_goal_frame(1, goal[:, :3])


