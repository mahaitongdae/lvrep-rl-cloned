import argparse
import os
import pickle as pkl

import gymnasium.wrappers.transform_reward
from tensorboardX import SummaryWriter
from datetime import datetime
from repr_control.utils import util, buffer
from repr_control.agent.sac import sac_agent
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.dpg import dpg_agent
# from define_problem import *
from gymnasium.envs.registration import register
import gymnasium
from repr_control.envs import ArticulateParking
import torch
import numpy as np
import yaml


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### parameter that
    parser.add_argument("--alg", default="mbdpg",
                        help="The algorithm to use. rfsac or sac.")
    parser.add_argument("--notes", type=str, default="change init dist",
                        help="The algorithm to use. rfsac or sac.")
    parser.add_argument("--horizon", default=240, type=int,
                        help="The algorithm to use. rfsac or sac.")
    parser.add_argument("--notes", default="change init dist", type=str,
                        help="The algorithm to use. rfsac or sac.")
    parser.add_argument("--env", default='parking',
                        help="Name your env/dynamics, only for folder names.")  # Alg name (sac, vlsac)
    parser.add_argument("--rf_num", default=512, type=int,
                        help="Number of random features. Suitable numbers for 2-dimensional system is 512, 3-dimensional 1024, etc.")
    parser.add_argument("--nystrom_sample_dim", default=8192, type=int,
                        help='The sampling dimension for nystrom critic. After sampling, take the maximum rf_num eigenvectors..')
    parser.add_argument("--device", default='cuda', type=str,
                        help="pytorch device, cuda if you have nvidia gpu and install cuda version of pytorch. "
                             "mps if you run on apple silicon, otherwise cpu.")

    parser.add_argument("--supervised", action='store_true',
                        help="add supervised learning.")
    parser.add_argument("--supervised_datasets", type=str, default="/datasets/2024-08-09_19-36-29/15_1.000_810000.pt",)
    parser.set_defaults(supervised=True)

    ### Parameters that usually don't need to be changed.
    parser.add_argument("--dir", default='main', type=str)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e4, type=float)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=512, type=int)  # Network hidden dims
    parser.add_argument("--hidden_depth", default=3, type=int)  # Latent feature dim
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--embedding_dim", default=-1, type=int)  # if -1, do not add embedding layer

    parser.add_argument("--use_nystrom", action='store_true')
    parser.add_argument("--use_random_feature", dest='use_nystrom', action='store_false')
    parser.set_defaults(use_nystrom=False)
    args = parser.parse_args()


    alg_name = args.alg
    exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    # setup example_results
    log_path = f'log/{alg_name}/{args.env}/{exp_name}'
    summary_writer = SummaryWriter(log_path + "/summary_files")

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = vars(args)
    env = gymnasium.make('ArticulateInfiniteHorizon-v0', horizon = args.horizon)
    eval_env = gymnasium.make('ArticulateInfiniteHorizon-v0', horizon = args.horizon)
    env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    eval_env = gymnasium.wrappers.RescaleAction(eval_env, min_action=-1, max_action=1)
    # kwargs.update({
    #     "state_dim": len(env.observation_space.low),
    #     "action_dim": len(env.action_space.low),
    #     "action_range": [env.action_space.low, env.action_space.high],
    #     # 'obs_space_high': np.clip(state_range[0], -3., 3.).tolist(),
    #     # 'obs_space_low': np.clip(state_range[1], -3., 3.).tolist(),  # in case of inf observation space
    # })

    # Initialize policy
    if args.alg == "sac":
        from repr_control.envs.models.articulate_model_fh import dynamics, reward, initial_distribution

        agent = sac_agent.ModelBasedSACAgent(6, 2, [[-1, -1], [1, 1]], dynamics, reward, initial_distribution, **kwargs)
    elif args.alg == "mbdpg":
        from repr_control.envs.models.articulate_model_fh import dynamics, rewards, initial_distribution
        agent = dpg_agent.ModelBasedDPGAgent(6, 2, [[-1, -1], [1, 1]], dynamics, rewards, initial_distribution, **kwargs)
    elif args.alg == "mbdpgqp":
        from repr_control.envs.models.articulate_model_fh import dynamics, rewards, initial_distribution

        agent = dpg_agent.ModelBasedQPDPGAgent(6, 2, [[-1, -1], [1, 1]], dynamics, rewards, initial_distribution,
                                             **kwargs)
    elif args.alg == 'rfsac':
        pass
    else:
        raise NotImplementedError("Algorithm not implemented.")

    replay_buffer = buffer.ReplayBuffer(len(env.observation_space.low),
                                        len(env.action_space.low), device=args.device)

    # register(id='custom-v0',
    #          entry_point='repr_control.envs:CustomEnv',
    #          max_episode_steps=max_step)


    # Evaluate untrained policy
    evaluations = []

    timer = util.Timer()

    # keep track of best eval model's state dict
    best_eval_reward = -1e6
    best_actor = None
    best_critic = None

    # save parameters

    with open(os.path.join(log_path, 'train_params.yaml'), 'w') as fp:
        yaml.dump(kwargs, fp, default_flow_style=False)

    if args.supervised:
        from repr_control.datasets.datasets import SupervisedParkingDataset
        from torch.utils.data import DataLoader
        cur_path = os.path.dirname(__file__)
        dataset = torch.load(cur_path + args.supervised_datasets)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        for supervised_t, supervised_data in enumerate(loader):
            info = agent.supervised_train(supervised_data)
            for key, value in info.items():
                summary_writer.add_scalar(f'info/{key}', value, supervised_t + 1)
            summary_writer.flush()

        actor = agent.actor.state_dict()
        torch.save(actor, os.path.join(log_path, 'actor_after_supervised.pth'))


    for t in range(int(args.max_timesteps + args.start_timesteps)):

        # episode_timesteps += 1


        info = agent.train(replay_buffer, batch_size=args.batch_size)

        if t % 10 == 0:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Rollout cost: {info['actor_loss']:.3f} Terminal cost: {info['terminal_cost']:.3f}, cost2go cost: {info['critic_loss']:.3f}")
            # Reset environment

        if info['terminal_cost'] > best_eval_reward:
            best_actor = agent.actor.state_dict()
            best_critic = agent.cost_to_go.state_dict()

            # save best actor/best critic
            torch.save(best_actor, log_path + "/best_actor.pth")
            torch.save(best_critic, log_path + "/best_cost_to_go.pth")

        if (t + 1) % 2 == 0:
            for key, value in info.items():
                if 'dist' not in key:
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), log_path + "/actor_last.pth")
    torch.save(agent.cost_to_go.state_dict(), log_path + "/cost_to_go_last.pth")