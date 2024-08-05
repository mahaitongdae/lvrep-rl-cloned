import argparse
import os
import pickle as pkl

from tensorboardX import SummaryWriter
from datetime import datetime
from repr_control.utils import util, buffer
from repr_control.agent.sac import sac_agent
from repr_control.agent.rfsac import rfsac_agent
# from define_problem import *
from repr_control.envs.models.articulate_model import *
from gymnasium.envs.registration import register
import gymnasium
import yaml
from repr_control.utils.buffer import Batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### parameter that
    parser.add_argument("--alg", default="sac",
                        help="The algorithm to use. rfsac or sac.")
    parser.add_argument("--env", default='custom_vec',
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
    parser.add_argument("--supervised_datasets", type=str, default="/datasets/2024-08-02_18-25-15/15_0.763_617760.pt",)
    parser.set_defaults(supervised=True)

    ### Parameters that usually don't need to be changed.
    parser.add_argument("--seed", default=0, type=int,
                        help='random seed.')  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float,
                        help='the number of initial steps that collects data via random sampled actions.')  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2000, type=int,
                        help='number of iterations as the interval to evaluate trained policy.')  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e4, type=float,
                        help='the total training time steps / iterations.')  # Max time steps to run environment
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--hidden_depth", default=3, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.05)  # Target network update rate
    parser.add_argument("--embedding_dim", default=-1, type=int)  # if -1, do not add embedding layer

    parser.add_argument("--use_nystrom", action='store_true')
    parser.add_argument("--use_random_feature", dest='use_nystrom', action='store_false')
    parser.set_defaults(use_nystrom=False)
    args = parser.parse_args()


    alg_name = args.alg
    exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    # setup example_results
    log_path = f'log/{alg_name}/{env_name}/{exp_name}'
    summary_writer = SummaryWriter(log_path + "/summary_files")
    print(f"Logging into {log_path}")

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = vars(args)
    kwargs.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_range": action_range,
        'obs_space_high': np.clip(state_range[0], -3., 3.).tolist(),
        'obs_space_low': np.clip(state_range[1], -3., 3.).tolist(),  # in case of inf observation space
    })

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif args.alg == 'rfsac':
        agent = rfsac_agent.CustomModelRFSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, **kwargs)
    else:
        raise NotImplementedError("Algorithm not implemented.")


    if args.env == 'custom_vec':
        from repr_control.envs.custom_env import CustomVecEnv

        env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            get_done=get_done,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=args.batch_size,
            device=torch.device(args.device), )
        eval_env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            get_done=get_done,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=args.batch_size,
            device=torch.device(args.device), )
    else:
        raise NotImplementedError("Environment not implemented.")

    # Evaluate untrained policy
    evaluations = []

    state, _ = env.reset()
    done = False
    episode_reward = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
    episode_timesteps = 0
    episode_num = 0
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

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            # action = env.action_space.sample()
            action = env.sample_action()
        else:
            action = agent.batch_select_action(state, explore=True)

        # Perform action
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        done = truncated

        batch = Batch(
			state=state,
			action=action,
			reward=reward,
			next_state=next_state,
			done=torch.zeros(size=(state.shape[0], 1), device=torch.device(args.device)),
		)

        state = next_state.clone()
        episode_reward += reward
        info = {}

        if t >= args.start_timesteps:
            if args.supervised and t < 1000:
                learn_policy = False
            else:
                learn_policy = True
            info = agent.batch_train(batch, learn_policy=learn_policy)


        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            avg_reward = episode_reward.mean().cpu().item()
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Average Reward: {avg_reward:.3f}")
            # Reset environment
            # info.update({'ep_len': episode_timesteps})
            state, _ =  env.reset()
            done = False
            episode_reward = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = util.batch_eval(agent, eval_env)
            evaluations.append(eval_ret)

            if t >= args.start_timesteps:
                info.update({'eval_ret': eval_ret})


            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()

                # save best actor/best critic
                torch.save(best_actor, log_path + "/best_actor.pth")
                torch.save(best_critic, log_path + "/best_critic.pth")

            best_eval_reward = max(evaluations)

        if (t + 1) % 500 == 0:
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
    torch.save(agent.critic.state_dict(), log_path + "/critic_last.pth")