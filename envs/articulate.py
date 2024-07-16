"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c 
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
import pygame
from pygame import gfxdraw
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from noisy_pend import angle_normalize

import envs


class ArticulateParking(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Articulated vehicles.
    state :
    1. x
    2. y
    3. theta
    4. xi, theta_1 - theta
    5. v
    6. delta

    action:
    1. accleration, between -1 and 1
    2. steering rate, takes 4 seconds to steer to self.delta_max

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    l = 4.9276  # tractor length
    R = 8.5349  # turning radius
    d1 = 15.8496  # trailer length
    delta_max = np.pi / 6
    dt = 0.05
    vehicle_length = 4.9276
    trailer_length = 15.8496
    vehicle_width = 1.9
    v_max = 0.6

    def __init__(self, render_mode: Optional[str] = None, task='swingup', noise_scale=0., eval=False):
        """
        task : swingup or balance
        """

        self.kinematics_integrator = "euler"

        self.noise_scale = noise_scale

        self.eval = eval

        # Angle at which to fail the episode
        self.task = task
        if task == 'balance':
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
        elif task == 'swingup':
            self.theta_threshold_radians = np.finfo(np.float32).max
        self.x_threshold = 2.4

        # is still within bounds.
        high = np.array(
            [
                20,
                10,
                np.pi,
                np.pi / 6,
                0.6,
                np.pi / 6
            ],
            dtype=np.float32,
        )
        self.state_bound = high

        self.action_space = spaces.Box(np.array([-1, -1]), np.ones([2, ]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_dim = 600
        # self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def angle_normalize(self, theta):
        return np.remainder(theta, 2 * np.pi)

    def ode(self, state, action):
        x, y, th0, dth, v, delta = state
        acc = action[0]
        delta_rate = action[1] * self.delta_max / 4
        normalized_steer = np.tan(delta) * self.R / self.l

        ds = np.array([
            v * np.cos(th0),
            v * np.sin(th0),
            v * normalized_steer / self.R,
            -1 * v * (self.d1 * normalized_steer + np.sin(dth) * self.R) / (self.R * self.d1),
            acc,
            delta_rate
        ])

        return ds

    def get_obs(self):
        return self.state

    def get_done(self):
        if np.any(np.abs(self.state) > self.state_bound):
            info = {'done_state': np.nonzero(np.abs(self.state) > self.state_bound)[0]}
            return True, info
        else:
            return False, {}

    def dynamics_step(self, action):
        ds = self.ode(self.state, action)
        # now we assume euler
        self.state = self.state + ds * self.dt
        self.state[-2] = np.clip(self.state[-2], - self.v_max, self.v_max)
        self.state[-1] = np.clip(self.state[-1], - self.delta_max, self.delta_max)


    def step(self, action):
        info = {}

        self.dynamics_step(action)

        terminated, done_info = self.get_done()
        info.update(done_info)
        rew_coeff = -1e-2 * np.array([1., 1., 10., 10.])

        if not terminated:
            # reward = 1.0
            reward = np.multiply(np.square(self.state[:4]), rew_coeff).sum()


        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = -100.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = -100.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        high = np.array(
            [
                10,
                3,
                np.pi / 6,
                np.pi / 6,
                0.0,
                0.0
            ],
            dtype=np.float32,
        )
        # low, high = utils.maybe_parse_reset_bounds(
        #     options,   # default low
        # )  # default high
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.np_random.uniform(low=-1 * high, high=high)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_dim, self.screen_dim)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 20.0
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        vehicle_length = self.vehicle_length * scale
        vehicle_width = 0.3 * self.vehicle_length * scale

        def draw_vehicle(vehicle_length, vehicle_width, state):
            """
            state: x, y, theta
            """
            l, r, t, b = 0, vehicle_length, vehicle_width / 2, -vehicle_width / 2
            vehicle_coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for c in vehicle_coords:
                c = pygame.math.Vector2(c).rotate_rad(state[2])
                c = (c[0] + scale * state[0] + offset, c[1] + scale * state[1] + offset)
                transformed_coords.append(c)
            gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
            # gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        draw_vehicle(vehicle_length, vehicle_width, self.state)


        trailer_dif = (pygame.math.Vector2((self.trailer_length, 0))
                       .rotate_rad(self.state[2] + self.state[3]))

        trailer_state = np.array([self.state[0] - trailer_dif[0],
                                  self.state[1] - trailer_dif[1],
                                  self.state[2] + self.state[3]])

        trailer_length = self.trailer_length * scale
        trailer_width = 0.3 * self.vehicle_length * scale
        draw_vehicle(trailer_length, trailer_width, trailer_state)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def test_env():
    import time
    env = ArticulateParking(render_mode='human')
    print(env.reset())
    env.render()
    import time
    for i in range(200):
        action = env.action_space.sample()
        state, _, _, _, _ = env.step(action)
        print(state[2], action)
        # time.sleep(0.1)
        env.render()


if __name__ == '__main__':
    test_env()
