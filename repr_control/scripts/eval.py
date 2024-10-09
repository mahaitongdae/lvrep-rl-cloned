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

    from repr_control.envs.models.articulate_model_fh import dynamics, evaluate_initial_states, terminal_constraints
    import copy
    actor = DeterministicActor(12, 2, kwargs['hidden_dim'], kwargs['hidden_depth'])
    actor.load_state_dict(torch.load(log_path + "/actor_after_supervised.pth"))

    init_state = evaluate_initial_states(15).float()
    obs = copy.deepcopy(init_state)
    for i in range(kwargs['horizon']):
        action = actor(torch.hstack([obs, init_state]))
        # noise = 0.1 * torch.randn_like(action)
        # action = torch.clamp(action + noise, min=-1, max=1)
        obs = dynamics(obs, action)
        # rewards += self.rewards(obs, action)
    terminal_constraint = terminal_constraints(obs)
    print(torch.max(terminal_constraint, dim=0))

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
                        , default='/home/naliseas-workstation/Documents/haitong/repr_control/lvrep-rl-cloned/log/mbdpgtc/parking/seed_0_2024-10-09-10-21-24')
    args = parser.parse_args()
    # eval(args.log_path)
    eval_mbdpg_agent(args.log_path)