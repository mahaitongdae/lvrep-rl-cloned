#this is a modified version - zhaolin

import math
import numpy as np
import time 
import torch
from envs.forward import DiffEnv
from envs.discretization import euler, runge_kutta4, runge_kutta4_cst_ctrl
from envs.torch_utils import smooth_relu


class Quadrotor(DiffEnv):
    def __init__(self, reg_speed=0.1, reg_ctrl=0.001, dt=0.008, horizon=180, stay_put_time=None, seed=0,
        init_state = np.asarray(([0., 0., 0.5, 0., 0., 0.,])), sigma = 0., max_speed = 8., euler = False):
        super(Quadrotor, self).__init__()
        # env parameters
        self.horizon, self.dt = horizon, dt
        self.dim_ctrl, self.dim_state = 2, 6
        self.init_time_iter = 0
        self.init_state= torch.from_numpy(init_state)
        self.euler = euler
        self.max_speed = max_speed
        if seed != 0:
            torch.manual_seed(seed)
            self.init_state[0] = self.init_state[0] + 1e0*torch.randn(1)

        #stay put time (?)

        self.stay_put_time = stay_put_time
        # print("stay ut time", self.stay_put_time)

        # cost parameters
        self.reg_speed, self.reg_ctrl = reg_speed, reg_ctrl
        # self.stay_put_time_start_iter = horizon-int(stay_put_time/dt) if self.stay_put_time is not None else horizon
        self.stay_put_time_start_iter = 0 # try start time iter = 0
        # physics parameters

        # rendering
        self.pole_transform = None

        #Noise
        self.sigma = sigma

    def __quad_action_preprocess(self, action):
        action = 0.075 * action + 0.075  # map from -1, 1 to 0.0 - 0.15
        # print(action)
        return action

    def quadrotor_f_star_6d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
        action = self.__quad_action_preprocess(action)
        def dot_states(states):
            dot_states = torch.hstack([states[1],
                                       1 / m * torch.multiply(torch.sum(action), torch.sin(states[4])),
                                       states[3],
                                       1 / m * torch.multiply(torch.sum(action), torch.cos(states[4])) - g,
                                       states[5],
                                       1 / 2 / Iyy * (action[1] - action[0]) * 0.025
                                       ])
            return dot_states

        k1 = dot_states(states)
        k2 = dot_states(states + dt / 2 * k1)
        k3 = dot_states(states + dt / 2 * k2)
        k4 = dot_states(states + dt * k3)

        return states + dt / 6 * (k1 + k2 + k3 + k4)

    def discrete_dyn(self, state, ctrl, time_iter):
        state = self.quadrotor_f_star_6d(state, ctrl)
        noise = np.random.normal(scale=self.sigma * self.dt, size=[6,])
        # noise = torch.multiply(torch.tensor([0., 1., 0., 1., 0., 1.]), torch.from_numpy(noise))
        noise = torch.multiply(torch.tensor([1., 0., 1., 0., 1., 0.]), torch.from_numpy(noise))
        # print("state shape", state.shape)
        return state + noise



    def costs(self, next_state, ctrl, time_iter):
        cost_next_state = self.cost_state(next_state, time_iter)
        return cost_next_state, self.cost_ctrl(ctrl)

    def cost_ctrl(self, ctrl):
        return - torch.sum((torch.log(1. - ctrl) +torch.log(1. + ctrl )))

    def cost_state(self, state, time_iter):
        if time_iter >= self.stay_put_time_start_iter:
            cost_state = state[0] ** 2 + (state[2] - 0.5) ** 2
        else:
            cost_state = torch.tensor(0.)
        return cost_state

    def reset(self, requires_grad=False):
        self.time_iter = self.init_time_iter
        self.state = self.init_state
        # print("self.state", self.state)
        self.state.requires_grad = requires_grad
        return self.state

    # def set_viewer(self):
    #     from envs import rendering
    #
    #     l = 2 * self.l
    #     self.viewer = rendering.Viewer(500, 500)
    #     self.viewer.set_bounds(-1.5 * l, 1.5 * l, -1.5 * l, 1.5 * l)
    #     rod = rendering.make_capsule(l, 0.1 * l)
    #     rod.set_color(0., 0., 0.)
    #     self.pole_transform = rendering.Transform()
    #     rod.add_attr(self.pole_transform)
    #     self.viewer.add_geom(rod)
    #
    # def render(self, title=None):
    #     if self.viewer is None:
    #         self.set_viewer()
    #     np_state = self.state.numpy()
    #     self.pole_transform.set_rotation(np_state[0] + np.pi/2)
    #     return self.viewer.render(title=title)




