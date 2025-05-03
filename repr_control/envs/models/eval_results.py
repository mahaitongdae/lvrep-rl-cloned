import pickle as pkl
import os
import argparse
from repr_control.agent.dpg import dpg_agent
from repr_control.envs.tractor_trailer_render import Renderer
from repr_control.agent.actor import DiagGaussianActor, DeterministicActor
from repr_control.utils.util import eval_policy
from repr_control.define_problem import *
from repr_control.envs.tractor_trailer_render import Renderer
from repr_control.envs.models.collision_checking.collsion_checking import collision_checking_tt
import gymnasium
from gymnasium.envs.registration import register
import yaml


def eval_mbdpg_agent(log_path, off_screen=False, actor='best'):
    try:
        with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:
            kwargs = pkl.load(f)
    except:
        with open(os.path.join(log_path, 'train_params.yaml'), 'r') as f:
            kwargs = yaml.safe_load(f)

    from repr_control.envs.models.articulate_model_cstr import dynamics, evaluate_initial_states, terminal_constraints, xy_rewards
    import copy
    # # from repr_control.envs.models.articulate_model_cstr import dynamics, xy_rewards, one_hot_rewards, \
    #         initial_distribution, terminal_constraints
    agent = dpg_agent.ModelBasedDPGAgentTerminalConstraintswithTrailer(
        state_dim=6,
        action_dim=2,
        action_range=[[-1, -1], [1, 1]],
        dynamics=dynamics,
        rewards=xy_rewards,
        initial_distribution=initial_distribution,
        terminal_constraints=terminal_constraints,
        action_noise=1.0,
        device='cpu'
    )
    actor = DeterministicActor(45, 2, kwargs['hidden_dim'], kwargs['hidden_depth'])
    if actor == 'best':
        actor.load_state_dict(torch.load(log_path + "/best_actor.pth", map_location='cpu'))
    elif actor == 'after_supervised':
        actor.load_state_dict(torch.load(log_path + "/actor_after_supervised.pth", map_location='cpu'))
    elif actor == 'last':
        actor.load_state_dict(torch.load(log_path + "/actor_last.pth", map_location='cpu'))
    
    
    init_obs = evaluate_initial_states(1)

    obs = init_obs
    init_state = init_obs[:, :6]
    
    x_init = init_obs[:, 6: 12]
    flattern_obstacle = init_obs[:, -33:-1]
    obstacle = flattern_obstacle.reshape(-1, 16, 2)
    trailer_length = init_obs[:, [-1]]

    state = copy.deepcopy(init_state)
    states = [state]
    rewards = torch.zeros([obs.shape[0]])
    dists = []
    dths = []

    with torch.no_grad():
        # start rollout
        for i in range(kwargs['horizon']):
            action = actor(agent.preprocess_observation(torch.hstack([state, x_init, flattern_obstacle, trailer_length]).float()))

            # if self.action_noise_std > 0:
            #     noise = self.action_noise_std * torch.randn_like(action)
            #     action = torch.clamp(action + noise, min=-1, max=1)

            # for o in obstacles:
            dist = collision_checking_tt(state, obstacle)
            dists.append(dist)
            dths.append(state[:, 3].unsqueeze(dim=1))

            # update state and obs
            state = dynamics(state, action,trailer_length.squeeze())
            states.append(state)
            obs = torch.hstack([state, x_init, flattern_obstacle, trailer_length])

            rewards += xy_rewards(state, action, x_init)

    terminal_constraint = terminal_constraints(state, x_init)
    
    states = torch.vstack(states)
    
    render = Renderer(vehicle_length=4.9276,
                        trailer_length=trailer_length.squeeze().item(),
                        save_video=True,
                        render_mode='rgb_array' if off_screen else 'human',)
    
    obstacles = torch.split(obstacle, 4, dim=1)
    obstacles = [o.squeeze().numpy() for o in obstacles]
    render.set_obstacles(obstacles)
    
    for state in states:
        render.set_state(state.numpy())
        render.render()
    render.save(dir=log_path)

    # handle maximum constraints
    # dists = torch.vstack(dists).T
    dists = torch.vstack(dists).T
    min_dists = torch.min(dists, dim=1)[0]
    dist_cstr = -1 * min_dists
    # dths = torch.hstack(dths)
    # min_dths = torch.max(torch.abs(dths), dim=1)[0]
    # dth_cstr = min_dths - torch.pi / 2
    all_cstr = torch.hstack([terminal_constraint,
                                dist_cstr.unsqueeze_(dim=1),
                                # dth_cstr.unsqueeze_(dim=1)
                                ])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='/home/naliseas-workstation/Documents/haitong/repr_control/lvrep-rl-cloned/repr_control/log/mbdpgtcobs/parking/seed_0_2025-04-27-21-03-52')
    args = parser.parse_args()

    eval_mbdpg_agent(args.log_path, actor='after_supervised')