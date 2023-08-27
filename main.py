import numpy as np
import torch
# import gym
import argparse
import os
import json

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent
from datetime import datetime

# from envs.noisy_pend import noisyPendulumEnv
from envs.env_helper import *

ENV_CONFIG = {'sin_input': True,              # fixed
              'reward_exponential': False,    # fixed
              'reward_scale': 1.,             # further tune
              'reward_type': 'energy',        # control different envs
              'theta_cal': 'sin_cos',         # fixed
              'noisy': False,                 # todo:depreciated
              'noise_scale': 0.               # should be same with sigma
              }


if __name__ == "__main__":
	
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=1, type=int)
  parser.add_argument("--alg", default="rfsac")                     # Alg name (sac, vlsac)
  parser.add_argument("--env", default="Pendubot-v0")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=1e4, type=float)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=2000, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=15e4, type=float)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=256, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--sigma", default = 1.,type = float) #noise for noisy environment
  parser.add_argument("--embedding_dim", default = -1,type =int) #if -1, do not add embedding layer
  parser.add_argument("--rf_num", default = 512, type = int)
  parser.add_argument("--nystrom_sample_dim", default=512, type=int,
                      help='sample dim, must be greater or equal rf num.')
  parser.add_argument("--learn_rf", default = "False") #make this a string (strange Python issue...) 
  parser.add_argument("--euler", default = "False")  #True if euler discretization to be used; otherwise use default OpenAI gym discretization
  parser.add_argument("--use_nystrom", default = "True")
  args = parser.parse_args()

  sigma = args.sigma
  euler = True if args.euler == "True" else False
  use_nystrom = True if args.use_nystrom == "True" else False

  ENV_CONFIG.update({'noisy': args.sigma, 'noise_scale': args.sigma})

  # initialize environments
  # env = gym.make(args.env)
  # eval_env = gym.make(args.env)

  if args.env == "Pendulum-v1":
    # env = noisyPendulumEnv(sigma =  sigma, euler = euler)
    # eval_env = noisyPendulumEnv(sigma = sigma, euler = euler)
    ENV_CONFIG.update({'reward_scale': 0.2, })
    env = env_creator_pendulum(ENV_CONFIG)
    ENV_CONFIG.update({'reward_scale': 1., })
    eval_env = env_creator_pendulum(ENV_CONFIG)
  elif args.env == 'Quadrotor2D-v1':
    ENV_CONFIG.update({'reward_scale': 1.,})
    env = env_creator(ENV_CONFIG)
    eval_env = env_creator(ENV_CONFIG)
  elif args.env == 'Pendubot-v0':
    eval_config = ENV_CONFIG.copy()
    eval_config.update({'reward_scale': 1., 'eval': True})
    eval_env = env_creator_pendubot(eval_config)
    ENV_CONFIG.update({'reward_scale': 10.})
    env = env_creator_pendubot(ENV_CONFIG)
  elif args.env == 'CartPoleContinuous-v0':
    ENV_CONFIG.update({'reward_scale': 1., })
    env = env_creator_cartpole(ENV_CONFIG)
    eval_env = env_creator_cartpole(ENV_CONFIG)
  env = Gymnasium2GymWrapper(env)
  eval_env = Gymnasium2GymWrapper(eval_env)
  # max_length = env._max_episode_steps
  # env.seed(args.seed)
  # eval_env.seed(args.seed)

  env_name = f'{args.env}_sigma_{args.sigma}_rew_scale_{ENV_CONFIG["reward_scale"]}'

  if args.env == 'Pendubot-v0':
    env_name = env_name + f'_reward_{ENV_CONFIG["reward_type"]}'

  alg_name = f'{args.alg}_nystrom_{use_nystrom}_rf_num_{args.rf_num}_sample_dim_{args.nystrom_sample_dim}'
  exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

  # setup log
  log_path = f'log/{env_name}/{alg_name}/{exp_name}'
  summary_writer = SummaryWriter(log_path+"/summary_files")

  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)


  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])

  if args.learn_rf == "False":
    learn_rf = False
  else:
    learn_rf = True

  # kwargs = {
  #   "discount": args.discount,
  #   "tau": args.tau,
  #   "hidden_dim": args.hidden_dim,
  #   "sigma": sigma,
  #   "rf_num": args.rf_num,
  #   "learn_rf": learn_rf,
  #   "use_nystrom": use_nystrom,
  #
  # }
  kwargs = vars(args)

  kwargs.update({
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    'obs_space_high': np.clip(env.observation_space.high, -10., 10.).tolist(),
    'obs_space_low': np.clip(env.observation_space.low, -10., 10.).tolist(),  # in case of inf observation space
    'obs_space_dim': env.observation_space.shape,
    'dynamics_type': args.env.split('-')[0],
    'dynamics_parameters': {
      'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    },
  })

  kwargs['dynamics_parameters'].update(ENV_CONFIG)

  # Initialize policy
  if args.alg == "sac":
    agent = sac_agent.SACAgent(**kwargs)
  elif args.alg == 'vlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = args.feature_dim
    agent = vlsac_agent.VLSACAgent(**kwargs)
  elif args.alg == 'rfsac':
    agent = rfsac_agent.RFSACAgent(**kwargs)
  
  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)

  # Evaluate untrained policy
  evaluations = [util.eval_policy(agent, eval_env)]

  state, done = env.reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  timer = util.Timer()

  #keep track of best eval model's state dict
  best_eval_reward = -1e6
  best_actor = None
  best_critic = None

  for t in range(int(args.max_timesteps)):
    
    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < args.start_timesteps:
      action = env.action_space.sample()
    else:
      action = agent.select_action(state, explore=True)



    # Perform action
    next_state, reward, done, _ = env.step(action) 
    # print("next state", next_state)
    # done_bool = float(done) if episode_timesteps < max_length else 0

    replay_buffer.add(state,action,next_state,reward,done)

    prev_state = np.copy(state)
    state = next_state
    episode_reward += reward
    
    # Train agent after collecting sufficient data
    if use_nystrom == True and t == args.start_timesteps: #init nystrom at the step training begins
      kwargs["replay_buffer"] = replay_buffer
      agent = rfsac_agent.RFSACAgent(**kwargs) #reinit agent is not ideal, temp fix

    if t >= args.start_timesteps:
      info = agent.train(replay_buffer, batch_size=args.batch_size)

    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} use_nystrom:{use_nystrom} rf_num:{args.rf_num} seed:{args.seed}")
      # Reset environment
      state, done = env.reset(), False
      # prev_state = np.copy(state)
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      steps_per_sec = timer.steps_per_sec(t+1)
      evaluation = util.eval_policy(agent, eval_env, eval_episodes=50)
      evaluations.append(evaluation)

      if t >= args.start_timesteps:
        info['evaluation'] = evaluation
        for key, value in info.items():
          summary_writer.add_scalar(f'info/{key}', value, t+1)
        summary_writer.flush()

      print('Step {}. Steps per sec: {:.4g}.'.format(t+1, steps_per_sec))

      if evaluation > best_eval_reward:
        best_actor = agent.actor.state_dict()
        best_critic = agent.critic.state_dict()

  summary_writer.close()

  print('Total time cost {:.4g}s.'.format(timer.time_cost()))

  #save best actor/best critic
  torch.save(best_actor, log_path+"/actor.pth")
  torch.save(best_critic, log_path+"/critic.pth")

  # save parameters
  kwargs.update({"action_space": None}) # action space might not be serializable
  with open(os.path.join(log_path, 'train_params.json'), 'w') as fp:
    json.dump(kwargs, fp, indent=2)
