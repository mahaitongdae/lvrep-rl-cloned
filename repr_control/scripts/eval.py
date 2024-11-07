import pickle as pkl
import os
import argparse
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.sac import sac_agent
from repr_control.agent.actor import DiagGaussianActor, DeterministicActor
from repr_control.utils.util import eval_policy
from repr_control.define_problem import *
import gymnasium
from gymnasium.envs.registration import register
import yaml
import torch

def eval(log_path, ):

    register(id='custom-v0',
             entry_point='repr_control.envs:CustomEnv',
             max_episode_steps=max_step)
    eval_env = gymnasium.make('custom-v0',
                   dynamics=dynamics,
                   rewards=rewards,
                   initial_distribution=initial_distribution,
                   state_range=state_range,
                   action_range=action_range,
                   sigma=sigma)
    eval_env = gymnasium.wrappers.RescaleAction(eval_env, min_action=-1, max_action=1)
    agent = get_controller(log_path)
    _, _, _, ep_rets = eval_policy(agent, eval_env, eval_episodes=50)

    return ep_rets

def eval_mbdpg_agent(log_path):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)

    from repr_control.envs.models.articulate_model_fh import dynamics, evaluate_initial_states, true_terminal_constraints
    import copy
    actor = DeterministicActor(6, 2, kwargs['hidden_dim'], kwargs['hidden_depth'])
    actor.load_state_dict(torch.load(log_path + "/actor_after_supervised.pth"))

    init_state = evaluate_initial_states(15).float()
    obs = copy.deepcopy(init_state)
    for i in range(kwargs['horizon']):
        # action = actor(torch.hstack([obs, init_state]))
        action = actor(obs)
        # noise = 0.1 * torch.randn_like(action)
        # action = torch.clamp(action + noise, min=-1, max=1)
        obs = dynamics(obs, action)
        # rewards += self.rewards(obs, action)
    terminal_constraint = true_terminal_constraints(obs)
    print(torch.max(terminal_constraint, dim=0))
    print(torch.sum(torch.all(terminal_constraint<0, dim=1)))

def plot_heatmap_mbdpg_agent(log_path, grid_size = 15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)

    from repr_control.envs.models.articulate_model_fh import dynamics, evaluate_initial_states, true_terminal_constraints
    import copy
    actor = DeterministicActor(6, 2, kwargs['hidden_dim'], kwargs['hidden_depth'])
    actor.load_state_dict(torch.load(log_path + "/actor_after_supervised.pth"))

    x = np.linspace(2., 5, grid_size)
    y = np.linspace(0.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    init_state = np.vstack([X_flat, Y_flat, np.zeros_like(X_flat),
                            np.zeros_like(X_flat),
                            np.zeros_like(X_flat),
                            np.zeros_like(X_flat) ]).T
    init_state = torch.from_numpy(init_state).float()
    obs = copy.deepcopy(init_state)
    for i in range(kwargs['horizon']):
        action = actor(obs)
        obs = dynamics(obs, action)
    terminal_constraint = true_terminal_constraints(obs).detach().numpy()
    print(terminal_constraint.shape)
    fig, axs = plt.subplots(2, 2, figsize = (15, 15))
    axs = axs.ravel()
    titles = ['terminal constraint: x', 'terminal constraint: y', 'terminal constraint: theta', 'terminal constraint: theta0 - theta1']
    x_ticklbs = [f"{num:.3g}" for num in x]
    y_ticklbs = [f"{num:.3g}" for num in y]
    for i in range(4):
        terminal_state = terminal_constraint[:, i].reshape((grid_size, grid_size))
        sns.heatmap(terminal_state, ax=axs[i], annot=True, annot_kws={"size": 6})
        axs[i].set_xticklabels(x_ticklbs, rotation=45)
        axs[i].set_yticklabels(y_ticklbs, rotation=45)
        axs[i].set_xlabel('init x')
        axs[i].set_ylabel('init y')
        axs[i].set_title(titles[i])
    fig.suptitle('Terminal constraint values, nonnegative value means satisfying constraints')
    plt.tight_layout()
    fig.savefig(os.path.join(log_path, 'terminal_heatmap.png'))

    # fig, axs = plt.subplots(1, 2, figsize = (15, 8))
    # sns.heatmap(X_flat.reshape(15, 15), ax=axs[0], annot=True) # , annot_kws={"size": 8}
    # sns.heatmap(Y_flat.reshape(15, 15), ax=axs[1], annot=True) # , annot_kws={"size": 8}
    # fig.savefig(os.path.join(log_path, 'grid_ref.png'))

def get_controller(log_path):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)
    if kwargs['alg'] == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif kwargs['alg'] == 'rfsac':
        agent = rfsac_agent.CustomModelRFSACAgent(dynamics_fn=dynamics, rewards_fn=rewards, **kwargs)
    else:
        raise NotImplementedError

    actor = DiagGaussianActor(obs_dim=kwargs['state_dim'],
                              action_dim=kwargs['action_dim'],
                              hidden_dim=kwargs['hidden_dim'],
                              hidden_depth=2,
                              log_std_bounds=[-5., 2.])

    actor.load_state_dict(torch.load(log_path + "/actor_last.pth"))
    agent.actor = actor
    agent.device = torch.device("cpu")
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str
                        , default='/home/naliseas-workstation/Documents/haitong/repr_control/lvrep-rl-cloned/log/mbdpgtc/parking/seed_0_2024-10-18-02-18-38')
    args = parser.parse_args()
    # eval(args.log_path)
    plot_heatmap_mbdpg_agent(args.log_path)