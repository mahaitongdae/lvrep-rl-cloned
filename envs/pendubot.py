#this is a modified version - zhaolin

import math
import numpy as np
import time 
import torch
from envs.forward import DiffEnv
from envs.discretization import euler, runge_kutta4, runge_kutta4_cst_ctrl
from envs.torch_utils import smooth_relu

device = torch.device('cpu')
class Pendubot(DiffEnv):
    def __init__(self, reg_speed=0.1, reg_ctrl=0.01, dt=0.05, horizon=40, stay_put_time=None, seed=0,
        init_state = np.asarray(([math.pi, 0.])), sigma = 0., max_speed = 8., euler = False):
        super(Pendubot, self).__init__()
        # env parameters
        self.horizon, self.dt = horizon, dt
        self.dim_ctrl, self.dim_state = 1, 4
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
        self.g, self.m, self.l, self.mu = 10, 1., 1., 0.01

        self.d1 = 0.089252
        self.d2 = 0.027630
        self.d3 = 0.023502
        self.d4 = 0.011204
        self.d5 = 0.002938

        # rendering
        self.pole_transform = None

        #Noise
        self.sigma = sigma

    def angle_normalize(self, th):
        return((th + np.pi) % (2 * np.pi) - np.pi)

    def pendubot_f_6d(self, states, action):
        dt = 0.05
        new_states = torch.empty_like(states, device=device)
        # states[:, 1], sin_theta1 = states[:, 0], states[:, 1]
        # cos_theta2, sin_theta2 = states[:, 2], states[:, 3]
        theta1_dot, theta2_dot = states[:, 2], states[:,3]
        theta1 = states[:, 0]
        theta2 = states[:, 1]
        new_theta1 = theta1 + dt * theta1_dot
        new_theta2 = theta2 + dt * theta2_dot
        new_states[:, 0] = new_theta1
        new_states[:, 1] = new_theta2
        # new_states[:, 2] = torch.cos(new_theta2)
        # new_states[:, 3] = torch.sin(new_theta2)

        d1 = 0.089252
        d2 = 0.027630
        d3 = 0.023502
        d4 = 0.011204
        d5 = 0.002938
        g = 9.81

        self.d4 = d4
        self.d5 = d5

        m11 = d1 + d2 + 2 * d3 * torch.cos(theta2)
        m21 = d2 + d3 * torch.cos(theta2)
        # m12 = d2 + d3 * torch.cos(theta2)
        m22 = d2

        mass_matrix = torch.empty((states.shape[0], 2, 2), device=device)
        mass_matrix[:, 0, 0] = m11
        mass_matrix[:, 0, 1] = m21
        mass_matrix[:, 1, 0] = m21
        mass_matrix[:, 1, 1] = m22

        self.mass_matrix = mass_matrix

        # mass_matrix = np.array([[m11, m12],
        #                         [m21, m22]])

        c_matrix = torch.empty((states.shape[0], 2, 2), device=device)
        c11 = -1. * d3 * torch.sin(theta2) * theta2_dot
        c12 = -d3 * torch.sin(theta2) * (theta2_dot + theta1_dot)
        c21 = d3 * torch.sin(theta2) * theta1_dot
        c22 = torch.zeros_like(theta1)
        c_matrix[:, 0, 0] = c11
        c_matrix[:, 0, 1] = c12
        c_matrix[:, 1, 0] = c21
        c_matrix[:, 1, 1] = c22

        g1 = d4 * torch.cos(theta2) * g + d5 * g * torch.cos(theta1 + theta2)
        g2 = d5 * torch.cos(theta1 + theta2) * g

        g_vec = torch.empty((states.shape[0], 2, 1), device=device)
        g_vec[:, 0, 0] = g1
        g_vec[:, 1, 0] = g2

        action = torch.hstack([action, torch.zeros_like(action)])[:, :, np.newaxis]
        acc = torch.linalg.solve(mass_matrix, action - torch.matmul(c_matrix, states[:, -2:][:, :, np.newaxis]) - g_vec)
        new_states[:, 2] = theta1_dot + dt * torch.squeeze(acc[:, 0])
        new_states[:, 3] = theta2_dot + dt * torch.squeeze(acc[:, 1])

        return new_states


    def discrete_dyn(self, state, ctrl, time_iter):
        next_state = self.pendubot_f_6d(torch.reshape(state, [1, -1]), torch.reshape(ctrl, [1, -1]))[0]
        noise = np.random.normal(scale=self.sigma * self.dt, size=[self.dim_state,])
        noise = torch.multiply(torch.tensor([0., 0., 1., 1.]), torch.from_numpy(noise))
        return next_state + noise

    def costs(self, next_state, ctrl, time_iter):
        cost_next_state = self.cost_state(next_state, time_iter)
        return cost_next_state, self.cost_ctrl(ctrl)

    def cost_ctrl(self, ctrl):
        return self.reg_ctrl * ctrl ** 2  -  (torch.log(0.5 - ctrl ) +torch.log(0.5 + ctrl ))

    def cost_state(self, state, time_iter):
        # print("hey start time iter", self.stay_put_time_start_iter)
        if time_iter >= self.stay_put_time_start_iter:
            # print("i here", time_iter)
            # cost_state = (self.angle_normalize(state[0])) ** 2 + self.reg_speed * state[1] ** 2
            cost_state = (torch.sin(state[0]) - 1.) ** 2 + torch.cos(state[0]) ** 2 + state[3] ** 2 + 0.01 * torch.sin(state[1]) ** 2 \
                + 0.01 * (torch.cos(state[1]) - 1.) ** 2 + 0.01 * state[3] ** 2
        else:
            cost_state = torch.tensor(0.)
        return cost_state






    def reset(self, requires_grad=False):
        self.time_iter = self.init_time_iter
        self.state = self.init_state
        # print("self.state", self.state)
        self.state.requires_grad = requires_grad
        return self.state

    def set_viewer(self):
        from envs import rendering

        l = 2 * self.l
        self.viewer = rendering.Viewer(500, 500)
        self.viewer.set_bounds(-1.5 * l, 1.5 * l, -1.5 * l, 1.5 * l)
        rod = rendering.make_capsule(l, 0.1 * l)
        rod.set_color(0., 0., 0.)
        self.pole_transform = rendering.Transform()
        rod.add_attr(self.pole_transform)
        self.viewer.add_geom(rod)

    def render(self, title=None):
        if self.viewer is None:
            self.set_viewer()
        np_state = self.state.numpy()
        self.pole_transform.set_rotation(np_state[0] + np.pi/2)
        return self.viewer.render(title=title)




