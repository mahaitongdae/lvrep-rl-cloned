import torch
import numpy as np


class DifferentiableDynamics(object):

    def __init__(self):
        pass

    def rollout(self, obs, action):
        pass


class Pendulum3D(DifferentiableDynamics):

    def __init__(self, g=10.0, m=1., l=1., max_a=2., max_speed=8., dt=0.05):
        super(Pendulum3D, self).__init__()
        self.g = torch.nn.Parameter(torch.tensor([g]))
        self.m = torch.nn.Parameter(torch.tensor([m]))
        self.l = torch.nn.Parameter(torch.tensor([l]))
        self.max_a = max_a
        self.max_speed = max_speed
        self.dt = dt


    def rollout(self, obs, action):
        th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
        thdot = obs[:, 2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action, -self.max_a, self.max_a)
        newthdot = thdot + (3. * self.g / (2 * self.l) * torch.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        new_states = torch.empty((obs.shape[0], 3))
        new_states[:, 0] = torch.cos(newth)
        new_states[:, 1] = torch.sin(newth)
        new_states[:, 2] = newthdot
        return new_states

    # def quadrotor_f_star_6d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
    #     dot_states = torch.empty_like(states)
    #     dot_states[:, 0] = states[:, 1]
    #     dot_states[:, 1] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4]))
    #     dot_states[:, 2] = states[:, 3]
    #     dot_states[:, 3] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g
    #     dot_states[:, 4] = states[:, 5]
    #     dot_states[:, 5] = 1 / 2 / Iyy * (action[:, 1] - action[:, 0])
    #
    #     new_states = states + dt * dot_states
    #
    #     return new_states
    #
    # def quadrotor_f_star_7d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
    #     new_states = torch.empty_like(states)
    #     new_states[:, 0] = states[:, 0] + dt * states[:, 1]
    #     new_states[:, 1] = states[:, 1] + dt * (
    #             1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4])))
    #     new_states[:, 2] = states[:, 2] + dt * states[:, 3]
    #     new_states[:, 3] = states[:, 3] + dt * (
    #             1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g)
    #     theta = torch.atan2(states[:, -2], states[:, -3])
    #     new_theta = theta + dt * states[:, 5]
    #     new_states[:, 4] = torch.cos(new_theta)
    #     new_states[:, 5] = torch.sin(new_theta)
    #     new_states[:, 6] = states[:, 6] + dt * (1 / 2 / Iyy * (action[:, 1] - action[:, 0]))
    #
    #     # new_states = states + dt * dot_states
    #
    #     return new_states
    #
    # def cartpole_f_4d(self, states, action, ):
    #     """
    #
    #     :param states: # x, x_dot, theta, theta_dot
    #     :param action: Force applied to the cart
    #     :return: new states
    #     """
    #     masscart = 1.0
    #     masspole = 0.1
    #     length = 0.5
    #     total_mass = masspole + masscart
    #     polemass_length = masspole * length
    #     dt = 0.02
    #     gravity = 9.81
    #     new_states = torch.empty_like(states)
    #     new_states[:, 0] = states[:, 0] + dt * states[:, 1]
    #     new_states[:, 2] = states[:, 2] + dt * states[:, 3]
    #     theta = states[:, 2]
    #     theta_dot = states[:, 3]
    #     costheta = torch.cos(theta)
    #     sintheta = torch.sin(theta)
    #     force = torch.squeeze(10. * action)
    #
    #     # For the interested reader:
    #     # https://coneural.org/florian/papers/05_cart_pole.pdf
    #     temp = 1. / total_mass * (
    #             force + polemass_length * theta_dot ** 2 * sintheta
    #     )
    #     thetaacc = (gravity * sintheta - costheta * temp) / (
    #             length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    #     )
    #     xacc = temp - polemass_length * thetaacc * costheta / total_mass
    #     new_states[:, 1] = states[:, 1] + dt * xacc
    #     new_states[:, 3] = theta_dot + dt * thetaacc
    #     return new_states
    #
    # def cartpole_f_5d(self, states, action, ):
    #     """
    #
    #     :param states: # x, x_dot, sin_theta, cos_theta, theta_dot
    #     :param action: Force applied to the cart
    #     :return: new states
    #     """
    #     masscart = 1.0
    #     masspole = 0.1
    #     length = 0.5
    #     total_mass = masspole + masscart
    #     polemass_length = masspole * length
    #     dt = 0.02
    #     gravity = 9.81
    #     new_states = torch.empty_like(states)
    #     new_states[:, 0] = states[:, 0] + dt * states[:, 1]
    #     costheta = states[:, -3]
    #     sintheta = states[:, -2]
    #     theta_dot = states[:, -1]
    #     theta = torch.atan2(sintheta, costheta)
    #     new_theta = theta + dt * theta_dot
    #     new_states[:, -3] = torch.cos(new_theta)
    #     new_states[:, -2] = torch.sin(new_theta)
    #     # new_states[:, 2] = states[:, 2] + dt * states[:, 3]
    #     # theta = states[:, 2]
    #
    #     force = torch.squeeze(10. * action)
    #
    #     # For the interested reader:
    #     # https://coneural.org/florian/papers/05_cart_pole.pdf
    #     temp = 1. / total_mass * (
    #             force + polemass_length * theta_dot ** 2 * sintheta
    #     )
    #     thetaacc = (gravity * sintheta - costheta * temp) / (
    #             length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    #     )
    #     xacc = temp - polemass_length * thetaacc * costheta / total_mass
    #     new_states[:, 1] = states[:, 1] + dt * xacc
    #     new_states[:, 4] = theta_dot + dt * thetaacc
    #     return new_states
    #
    # def pendubot_f_6d(self, states, action):
    #     import torch
    #     dt = 0.05
    #     new_states = torch.empty_like(states)
    #     cos_theta1, sin_theta1 = states[:, 0], states[:, 1]
    #     cos_theta2, sin_theta2 = states[:, 2], states[:, 3]
    #     theta1_dot, theta2_dot = states[:, 4], states[:, 5]
    #     theta1 = torch.atan2(sin_theta1, cos_theta1)
    #     theta2 = torch.atan2(sin_theta2, cos_theta2)
    #     new_theta1 = theta1 + dt * theta1_dot
    #     new_theta2 = theta2 + dt * theta2_dot
    #     new_states[:, 0] = torch.cos(new_theta1)
    #     new_states[:, 1] = torch.sin(new_theta1)
    #     new_states[:, 2] = torch.cos(new_theta2)
    #     new_states[:, 3] = torch.sin(new_theta2)
    #
    #     d1 = 0.089252
    #     d2 = 0.027630
    #     d3 = 0.023502
    #     d4 = 0.011204
    #     d5 = 0.002938
    #     g = 9.81
    #
    #     self.d4 = d4
    #     self.d5 = d5
    #
    #     m11 = d1 + d2 + 2 * d3 * torch.cos(theta2)
    #     m21 = d2 + d3 * torch.cos(theta2)
    #     # m12 = d2 + d3 * torch.cos(theta2)
    #     m22 = d2
    #
    #     mass_matrix = torch.empty((states.shape[0], 2, 2))
    #     mass_matrix[:, 0, 0] = m11
    #     mass_matrix[:, 0, 1] = m21
    #     mass_matrix[:, 1, 0] = m21
    #     mass_matrix[:, 1, 1] = m22
    #
    #     self.mass_matrix = mass_matrix
    #
    #     # mass_matrix = np.array([[m11, m12],
    #     #                         [m21, m22]])
    #
    #     c_matrix = torch.empty((states.shape[0], 2, 2))
    #     c11 = -1. * d3 * np.sin(theta2) * theta2_dot
    #     c12 = -d3 * np.sin(theta2) * (theta2_dot + theta1_dot)
    #     c21 = d3 * np.sin(theta2) * theta1_dot
    #     c22 = torch.zeros_like(theta1)
    #     c_matrix[:, 0, 0] = c11
    #     c_matrix[:, 0, 1] = c12
    #     c_matrix[:, 1, 0] = c21
    #     c_matrix[:, 1, 1] = c22
    #
    #     g1 = d4 * torch.cos(theta2) * g + d5 * g * torch.cos(theta1 + theta2)
    #     g2 = d5 * torch.cos(theta1 + theta2) * g
    #
    #     g_vec = torch.empty((states.shape[0], 2, 1))
    #     g_vec[:, 0, 0] = g1
    #     g_vec[:, 1, 0] = g2
    #
    #     action = torch.hstack([action, torch.zeros_like(action)])[:, :, np.newaxis]
    #     acc = torch.linalg.solve(mass_matrix, action - torch.matmul(c_matrix, states[:, -2:][:, :, np.newaxis]) - g_vec)
    #     new_states[:, 4] = theta1_dot + dt * torch.squeeze(acc[:, 0])
    #     new_states[:, 5] = theta2_dot + dt * torch.squeeze(acc[:, 1])
    #
    #     return new_states
    #
    # def _get_energy_error(self, obs, action, ke=1.5):
    #     assert self.q_net.dynamics_type == 'Pendubot'
    #     dot_theta = obs[:, -2:][:, :, np.newaxis]  # batch, 2, 1
    #     dot_theta_t = obs[:, -2:][:, np.newaxis]  # batch, 1, 2
    #     cos_theta1, sin_theta1 = obs[:, 0], obs[:, 1]
    #     cos_theta2, sin_theta2 = obs[:, 2], obs[:, 3]
    #     sin_theta1_plus_theta2 = torch.multiply(sin_theta1, cos_theta2) + torch.multiply(cos_theta1, sin_theta2)
    #
    #     kinetic_energy = torch.squeeze(torch.matmul(torch.matmul(dot_theta_t, self.mass_matrix), dot_theta))
    #     potential_energy = self.d4 * 9.81 * sin_theta1 + self.d5 * 9.81 * sin_theta1_plus_theta2
    #     energy_on_top = (self.d4 + self.d5) * 9.81
    #     energy_error = kinetic_energy + potential_energy - energy_on_top
    #
    #     return ke * energy_error ** 2
    #
    # def angle_normalize(self, th):
    #     return ((th + np.pi) % (2 * np.pi)) - np.pi
