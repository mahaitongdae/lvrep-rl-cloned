import numpy as np
import torch
import gym
import argparse
import os

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent

from our_env.noisy_pend import noisyPendulumEnv

from agent.rfsac.rfsac_agent import RFVCritic
from agent.sac.actor import DiagGaussianActor
from agent.rfsac.rfsac_agent import RFSACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(env, log_path, agent, alg, alpha, rf_num = 512, learn_rf = False, max_steps=200, state_dim=None):
    np.random.seed(0)
    n_init_states = 50
    low_arr = [-np.pi, -1.]
    high_arr = [np.pi, 1.]
    # init_states = np.vstack([np.random.uniform(low_arr[0], high_arr[0], n_init_states),
    #                          np.random.uniform(low_arr[1], high_arr[1], n_init_states)])
    init_states = np.random.uniform(low=low_arr, high=high_arr, size=(n_init_states, 2))
    init_obs = np.vstack([np.cos(init_states[:, 0]), np.sin(init_states[:, 0]), init_states[:, 1]]).T
    init_actions = agent.select_action(init_obs)
    init_dist = agent.actor(torch.tensor(init_obs).float().to(device))
    init_log_prob = init_dist.log_prob(torch.tensor(init_actions).to(device))
    if alg == 'rfsac':
        init_rew = agent.get_reward(torch.tensor(init_obs).to(device),
                                    torch.tensor(init_actions).to(device)) - alpha * init_log_prob
        v1, v2 = agent.critic(
            agent.dynamics_step(torch.tensor(init_obs).to(device), torch.tensor(init_actions).to(device)))
        all_init_agent_values = torch.min(v1, v2)
        init_qs = (init_rew + all_init_agent_values).detach().clone().cpu().numpy().squeeze()
    elif alg == 'sac':
        q1, q2 = agent.critic(torch.tensor(init_obs).float().to(device),
                              torch.tensor(init_actions).float().to(device))
        init_qs = torch.min(q1, q2).detach().clone().cpu().numpy().squeeze()
    all_rewards = []
    u_list = np.empty(max_steps)
    for i in np.arange(n_init_states):
        init_state = init_states[i]
        state = env.reset(init_state=init_state)
        eps_reward = 0
        gamma = 1.
        for t in range(max_steps):
            action = agent.select_action(np.array(state))
            dist = agent.actor(torch.tensor(state).to(device))
            log_prob = dist.log_prob(torch.tensor(action).to(device))
            state, reward, done, _ = env.step(action)
            eps_reward += gamma * (0.1 * reward - alpha * log_prob.cpu().item())
            gamma *= 0.99
            # if i == 4:
            #     print(gamma, reward, eps_reward)
            u_list[t] = action
        # eps_reward += agent.critic(agent.dynamics_step(torch.tensor()))
        all_rewards.append(eps_reward)


    # print(init_states)
    print(init_qs)
    print(all_rewards)

    return init_qs, all_rewards

    # env.visualize(init_state=init_states[i], cmd=u_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=0, type=int)
    parser.add_argument("--alg", default="sac")  # Alg name (sac, vlsac,rfsac)
    parser.add_argument("--env", default="Pendulum-v1")  # Environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=float)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=512, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    parser.add_argument("--sigma", default=0., type=float)  # noise for noisy environment
    parser.add_argument("--rand_feat_num", default=512, type=int)
    parser.add_argument("--learn_rf", default="False")  # string indicating if learn_rf is false or no
    parser.add_argument("--dir_name", default="2023-04-10-02-34-45")
    # 2023-04-10-02-08-02 for reproduing Zhaolin's,
    # 2023-04-12-10-39-00 for 1024 rfdim
    # 2023-04-10-02-34-45 for sac
    # 2023-04-17-00-30-06 new rfsac 512 alpha: 0.02458

    dict = {'rfsac': '/home/mht/PycharmProjects/lvrep-rl-cloned/exp/Pendulum-v1/rfsac/0/0/T=200000.0/rf_num=512/learn_rf=False/sigma=0.0/2023-04-17-00-30-06',
            'sac': '/home/mht/PycharmProjects/lvrep-rl-cloned/exp/Pendulum-v1/sac/0/0/T=100000.0/2023-04-10-02-34-45',
            }
    args = parser.parse_args()

    sigma = args.sigma
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_length = env._max_episode_steps
    print("hey", args.learn_rf)
    learn_rf = True if args.learn_rf == "True" else False
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "sigma": sigma,
        "rand_feat_num": args.rand_feat_num,
        "learn_rf": learn_rf
    }

    # Initialize policy


    if args.env == "Pendulum-v1":
        env = noisyPendulumEnv(sigma=sigma)
        for key, value in dict.items():
            if key == "sac":
                agent = sac_agent.SACAgent(**kwargs)
                log_path = value
                agent.actor.load_state_dict(torch.load(log_path + "/actor.pth"))
                agent.critic.load_state_dict(torch.load(log_path + "/critic.pth"))
                learnqsac, realq = main(env, log_path, agent, key, alpha = 0.0, rf_num=args.rand_feat_num, learn_rf=learn_rf,
                                    state_dim=state_dim)
            elif key == 'rfsac':
                agent = rfsac_agent.RFSACAgent(**kwargs)
                log_path = value
                actor = DiagGaussianActor(obs_dim=3, action_dim=1, hidden_dim=args.hidden_dim, hidden_depth=2,
                                          log_std_bounds=[-5., 2.]).to(agent.device)
                critic = RFVCritic(sigma=sigma, rand_feat_num=args.rand_feat_num, learn_rf=learn_rf).to(agent.device)
                actor.load_state_dict(torch.load(log_path + "/actor.pth"))
                critic.load_state_dict(torch.load(log_path + "/critic.pth"))
                agent.actor = actor
                agent.critic = critic
                learnqrfsac, realq = main(env, log_path, agent, key, alpha = 0.02458, rf_num=args.rand_feat_num, learn_rf=learn_rf,
                                        state_dim=state_dim)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(50), learnqrfsac, label='random feature')
        plt.plot(range(50), learnqsac, label='mlp')
        plt.plot(range(50), realq, label='true q value')
        plt.xlabel('random state index')
        plt.ylabel('Q value')
        plt.legend()
        plt.show()


