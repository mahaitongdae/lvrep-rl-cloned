import gymnasium
import numpy as np
from gymnasium.wrappers import TransformReward
try:
    import safe_control_gym
    from safe_control_gym.utils.configuration import ConfigFactory
    from safe_control_gym.utils.registration import make, register
except:
    pass



ENV_CONFIG = {'sin_input': True,
              'reward_exponential': False,
              'reward_scale': 10.,
              'reward_type' : 'energy',
              'theta_cal': 'sin_cos',
              'noisy': False,
              'noise_scale': 0.
              }

class TransformTriangleObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # low = np.array([
        #     -env.x_threshold, -np.finfo(np.float32).max,
        #     env.GROUND_PLANE_Z, -np.finfo(np.float32).max,
        #     -1., -1., -np.finfo(np.float32).max
        # ])
        # high = np.array([
        #     env.x_threshold, np.finfo(np.float32).max,
        #     env.z_threshold, np.finfo(np.float32).max,
        #     1., 1., np.finfo(np.float32).max
        # ])
        low = env.observation_space.low
        high = env.observation_space.high
        transformed_low = np.hstack([low[:-2], [-1., -1.,], low[-1:]])
        transformed_high = np.hstack([high[:-2], [1., 1.,], high[-1:]])

        self.observation_space = gymnasium.spaces.Box(low=transformed_low, high=transformed_high, dtype=np.float32)

    def observation(self, observation):
        '''
        transfer observations. We assume that the last two observations is the angle and angular velocity.
        '''
        theta = observation[-2]
        sin_cos_theta = np.array([np.cos(theta), np.sin(theta)])
        theta_dot = observation[-1:]
        return np.hstack([observation[:-2], sin_cos_theta, theta_dot])

class TransformDoubleTriangleObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # low = np.array([
        #     -env.x_threshold, -np.finfo(np.float32).max,
        #     env.GROUND_PLANE_Z, -np.finfo(np.float32).max,
        #     -1., -1., -np.finfo(np.float32).max
        # ])
        # high = np.array([
        #     env.x_threshold, np.finfo(np.float32).max,
        #     env.z_threshold, np.finfo(np.float32).max,
        #     1., 1., np.finfo(np.float32).max
        # ])
        low = env.observation_space.low
        high = env.observation_space.high
        transformed_low = np.hstack([[-1., -1., -1., -1.,], low[-2:]])
        transformed_high = np.hstack([[1., 1., 1., 1., ], high[-2:]])

        self.observation_space = gymnasium.spaces.Box(low=transformed_low, high=transformed_high, dtype=np.float32)

    def observation(self, observation):
        theta1 = observation[0]
        theta2 = observation[1]
        sin_cos_theta1 = np.array([np.cos(theta1), np.sin(theta1)])
        sin_cos_theta2 = np.array([np.cos(theta2), np.sin(theta2)])
        theta_dot = observation[-2:]
        return np.hstack([sin_cos_theta1, sin_cos_theta2, theta_dot])

class NoisyObservationWrapper(gymnasium.Wrapper):

    def __init__(self, env, noise_scale):
        super().__init__(env)
        # np.random.seed(seed)
        self.noise_scale = noise_scale

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.env.state[0] = self.env.state[0] + self.np_random.normal(scale=self.noise_scale) * self.env.dt
        return self._get_obs(), reward, done, terminated, info

class Gymnasium2GymWrapper(gymnasium.Wrapper):

    def __int__(self, env):
        super(Gymnasium2GymWrapper, self).__int__(env)

    def seed(self, seed):
        self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        return obs, reward, done or terminated, info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

def env_creator(env_config):
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./quad_2d_env_config/stabilization.yaml'])
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)
    if env_config.get('sin_input'):
        trans_rew_env = TransformReward(env, lambda r: env_config.get('reward_scale') * r)
        return TransformTriangleObservationWrapper(trans_rew_env)
    else:
        return TransformReward(env, lambda r: env_config.get('reward_scale') * r)

def env_creator_pendulum(env_config):
    env = gymnasium.make('Pendulum-v1')
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(env_config.get('reward_scale') * r))
    else:
        env = TransformReward(env, lambda r: env_config.get('reward_scale') * r)
    if env_config.get('noisy'):
        env = NoisyObservationWrapper(env, noise_scale=env_config.get('noise_scale', 1))
    return env

def env_creator_cartpole(env_config):
    from gymnasium.envs.registration import register

    register(id='CartPoleContinuous-v0',
             entry_point='envs:CartPoleEnv',
             max_episode_steps=300)
    env = gymnasium.make('CartPoleContinuous-v0') #, render_mode='human'
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(r))
    if env_config.get('sin_input'):
        return TransformTriangleObservationWrapper(env)
    else:
        return env

def env_creator_pendubot(env_config):
    from gymnasium.envs.registration import register
    reward_scale_pendubot = env_config.get('reward_scale')
    noisy = env_config.get('noisy')
    noise_scale = env_config.get('noise_scale')
    register(id='Pendubot-v0',
             entry_point='envs:PendubotEnv',
             max_episode_steps=200)
    env = gymnasium.make('Pendubot-v0',
                         noisy=noisy,
                         noisy_scale=noise_scale,
                         reward_type=env_config.get('reward_type'),
                         theta_cal=env_config.get('theta_cal')
                         ) #, render_mode='human'
    env = TransformReward(env, lambda r: reward_scale_pendubot * r)
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(r))
    if env_config.get('sin_input'):
        return TransformDoubleTriangleObservationWrapper(env)
    else:
        return env

