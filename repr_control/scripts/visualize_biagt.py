import pickle as pkl
import numpy as np
# from repr_control.envs.env_helper import *
from repr_control.envs import ArticulateParkingInfiniteHorizon
import argparse
import os
import torch
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.sac import sac_agent
from repr_control.agent.dpg import dpg_agent
from repr_control.agent.actor import DiagGaussianActor
from repr_control.utils.util import eval_policy
import gymnasium
import yaml
import matplotlib.pyplot as plt

def eval(log_path, ):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)
    kwargs['device'] = 'cpu'

    # eval_env = gymnasium.make('ArticulateInfiniteHorizon-v0', render_mode='human', horizon=500, save_video=True)
    eval_env = ArticulateParkingInfiniteHorizon(render_mode='human', horizon=250, save_video=True)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'qpsac':
        agent = sac_agent.QPSACAgent(**kwargs)
    elif kwargs['alg'] == "mbdpg":
        agent = dpg_agent.DPGAgent(state_dim=6, action_dim=2, action_range = [[-1, -1], [1, 1]], **kwargs)
    else:
        raise NotImplementedError

    agent.actor.load_state_dict(torch.load(log_path+"/best_actor.pth", map_location=torch.device('cpu')))
    # agent.actor = actor
    agent.device = torch.device("cpu")

    _, _, _, ep_rets = eval_biagt(agent, eval_env,
                                  eval_episodes=1,
                                  render=True,
                                  state=np.array([ 2.   ,       1.5    ,     0.2 , -0.2 , 0.       ,   0.        ])
                                  ) # seed=3
                                  # seed=5)
    eval_env.close()

    return ep_rets

def eval_biagt(policy, eval_env, eval_episodes=100, render=False, seed=0, state=None, plot=True):
    """
    Eval a policy
    """
    ep_rets = []
    avg_len = 0.
    states = []
    actions = []
    import time
    for i in range(eval_episodes):
        ep_ret = 0.
        # eval_env.seed(i)
        if state is not None:
            state, _ = eval_env.reset(options={'state': state})
        else:
            state, _ = eval_env.reset(seed=seed) # options={"state": np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        done = False
        states.append(state)
        # print("eval_policy state", state)
        while not done:
            start = time.time()
            action = policy.select_action(np.array(state))
            print(time.time() - start)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
            avg_len += 1
            actions.append(action)
            states.append(state)
            if render:
                eval_env.render()
        ep_rets.append(ep_ret)

    avg_ret = np.mean(ep_rets)
    std_ret = np.std(ep_rets)
    avg_len /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: avg eplen {avg_len}, avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}")
    print("---------------------------------------")

    states = np.vstack(states)
    actions = np.vstack(actions)

    if plot:
        n = 6
        m = 2
        # Plot state and control trajectories
        fig, ax = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
        plt.subplots_adjust(wspace=0.45)
        labels_s = (r'$x(t)$', r'$y(t)$', r'$\theta_0{x}(t)$', r'$\theta_0 - \theta_1(t)$', r'v(t)', 'delta(t)')
        labels_u = (r'$a(t)$', r'$ddelta(t)$')
        for i in range(n):
            ax[i].plot(states[:, i])
            ax[i].axhline(0.0, linestyle='--', color='tab:orange')
            ax[i].set_xlabel(r'$t$')
            ax[i].set_title(labels_s[i])
        for i in range(m):
            ax[n + i].plot(actions[:, i])
            # ax[n + i].axhline(u_max[i], linestyle='--', color='tab:orange')
            # ax[n + i].axhline(-u_max[i], linestyle='--', color='tab:orange')
            ax[n + i].set_xlabel(r'$t$')
            ax[n + i].set_title(labels_u[i])
        plt.show()
    return avg_len, avg_ret, std_ret, ep_rets

def plot_cost_to_go(log_path):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)
    kwargs['device'] = 'cpu'

    # eval_env = gymnasium.make('ArticulateInfiniteHorizon-v0', render_mode='human', horizon=500, save_video=True)
    eval_env = ArticulateParkingInfiniteHorizon(render_mode='human', horizon=250, save_video=True)
    kwargs['action_space'] = eval_env.action_space
    kwargs.update({'eval': True})
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'qpsac':
        agent = sac_agent.QPSACAgent(**kwargs)
    elif kwargs['alg'] == "mbdpg":
        from repr_control.envs.models.articulate_model_fh import dynamics, rewards, initial_distribution
        agent = dpg_agent.ModelBasedDPGAgent(dynamics = dynamics, rewards=rewards, initial_distribution=initial_distribution,
                                             state_dim=6, action_dim=2, action_range = [[-1, -1], [1, 1]], **kwargs)
    else:
        raise NotImplementedError

    agent.cost_to_go.load_state_dict(torch.load(log_path+"/best_cost_to_go.pth", map_location=torch.device('cpu')))
    # agent.actor = actor
    x_range = np.linspace(2, 5, 100)
    y_range = np.linspace(0.0, 1.5, 100)

    # Create a meshgrid of x and y values
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    grid_shape = x_grid.shape

    # Flatten the grid to create input pairs
    input_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
    theta_0 = np.pi / 12 * np.ones((input_grid.shape[0], 1))
    input_full = np.hstack([input_grid, theta_0, -1 * theta_0, np.zeros((input_grid.shape[0], 2))])

    # Convert the input grid to a PyTorch tensor
    input_full_tensor = torch.tensor(input_full, dtype=torch.float32)

    # Pass the grid through the model to get predictions
    with torch.no_grad():
        output_grid = agent.cost_to_go(input_full_tensor).numpy()

    # Reshape the output to match the grid shape
    output_grid = output_grid.reshape(grid_shape)

    # Plot the results using a contour plot
    plt.figure(figsize=(8, 4))
    contour = plt.contourf(x_grid, y_grid, output_grid, levels=100, cmap='viridis')
    plt.colorbar(contour)
    plt.title('Value Function Visualization direction -15 degree')
    plt.ylim(0.0, 1.5)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    plot_cost_to_go("/Users/mahaitong/Code/repr_control/repr_control/log/mbdpg/parking/seed_0_2024-08-27-22-46-35")