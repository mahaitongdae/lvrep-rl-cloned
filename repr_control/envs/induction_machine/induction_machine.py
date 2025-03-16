from typing import List
import gymnasium
import numpy as np
from gymnasium.spaces.box import Box


class InductionMachineEnv(gymnasium.Env):
    pole_pair: float = 2
    rotor_inertia: float = 0.0163
    stator_resist: float = 0.439
    rotor_resist: float = 0.410
    mutual_induct: float = 60.1e-3
    stator_leak_induct: float = 1.4e-3
    rotor_leak_induct: float = 1.8e-3

    # discretization
    Ts = 1 / 10000  # 10kHz

    # goal
    omega_star: float = 5.
    phi_star: float = 0.5
    tl_star: float = 1.

    # cost
    Q: List[float] = [1., 0.1, 1000., ]
    R: List[float] = [0.001, 0.001]

    p = pole_pair
    Lm = mutual_induct
    Lr = Lm + rotor_leak_induct
    Ls = Lm + stator_leak_induct
    Rr = rotor_resist
    Rs = stator_resist
    J = rotor_inertia

    sigma = Ls * (1 - Lm * Lm / (Ls * Lr))
    beta = Lm / (sigma * Lr)
    # mu = 3*Lm*p/(2*Lr)       # MERL formula
    mu = Lm * p / Lr  # ATC formula
    mu2 = mu ** 2
    alpha = Rr / Lr
    gamma = (Rs / sigma) + alpha * Lm * beta

    # equilibirum state
    id_ref = phi_star / Lm
    iq_ref = tl_star / (mu * phi_star)
    phid_ref = phi_star
    phid_ref2 = phid_ref ** 2
    phid_ref3 = phid_ref ** 3
    phiq_ref = 0.
    omega_ref = omega_star

    def __init__(self, raw_action = False):
        self.observation_space = Box(np.array([-np.inf] * 5, ),
                                     np.array([np.inf] * 5), )
        self.action_space = Box(-1 * np.ones([2, ]),
                                np.ones([2, ]))
        # self.action_space = Box(np.array([-np.inf] * 2), np.array([np.inf] * 2),)
        self.action_scale = np.array([300, 300])
        self.equilibirum_state = np.array([
            self.id_ref, self.iq_ref, self.phid_ref, self.phiq_ref, self.omega_ref
        ], dtype=np.float32)

        self.equilibirum_action = self.sigma * np.array([
            (self.gamma * self.id_ref - self.alpha * self.beta * self.phid_ref -
             self.tl_star / self.mu2 / self.phid_ref3 * (
                         self.mu * self.omega_ref * self.phid_ref2 + self.Lm * self.alpha * self.tl_star)),
            (self.gamma * self.tl_star / self.mu / self.phid_ref + self.beta * self.omega_ref * self.phid_ref +
             self.id_ref / self.mu / self.phid_ref2 * (
                         self.mu * self.omega_ref * self.phid_ref2 + self.Lm * self.alpha * self.tl_star)),
        ])

        self.raw_action = raw_action

    def reset(self, *, seed = None, options = {}):
        if options is None or 'init_state' not in options.keys():
            self.state = np.array([0, 0, 1e-2, 0, 0], dtype=np.float32)
        else:
            assert options['init_state'].shape == (5,)
            self.state = options['init_state']
        self.current_step = 0
        return self.state, {}

    def get_terminated(self):
        if (np.abs(self.state) > 1000.).any():
            return True
        else:
            return False

    def step(self, action):
        if not self.raw_action:
            action_after_preprocess = self.preprocess_action(action)
        else:
            action_after_preprocess = action
        # action_after_preprocess = action
        dx = self.dynamics(action_after_preprocess)
        self.state = self.state + dx * self.Ts
        reward = self.compute_reward(self.state, action_after_preprocess)
        self.current_step += 1
        truncated = True if self.current_step == 1000 else False
        terminated = self.get_terminated()
        if terminated:
            reward -= 500.
        info = {}

        return self.state, reward, terminated, truncated, info

    def preprocess_action(self, action):
        return (action * self.action_scale + self.equilibirum_action).astype(np.float32)

    def compute_reward(self, state, action):
        state_error = (self.state - self.equilibirum_state)[[0, 1, 4]]
        state_cost = self.Q * state_error ** 2
        action_error = action - self.equilibirum_action
        action_cost = 0.001 * action_error ** 2
        return - 1 * (state_cost.sum() + action_cost.sum())  # 10. * 10000

    def dynamics(self, action_after_preprocess):
        i_ds = self.state[0]
        i_qs = self.state[1]
        phi_dr = self.state[2]
        phi_qr = self.state[3]
        omega = self.state[4]

        u_ds = action_after_preprocess[0]
        u_qs = action_after_preprocess[1]

        omega_1 = omega + self.alpha * self.Lm * i_qs / phi_dr

        dx1 = -self.gamma * i_ds + omega_1 * i_qs + self.beta * (
                    self.alpha * phi_dr + omega * phi_qr) + u_ds / self.sigma
        dx2 = -self.gamma * i_qs - omega_1 * i_ds + self.beta * (
                    self.alpha * phi_qr - omega * phi_dr) + u_qs / self.sigma
        dx3 = -self.alpha * phi_dr + (omega_1 - omega) * phi_qr + self.alpha * self.Lm * i_ds
        dx4 = -self.alpha * phi_qr - (omega_1 - omega) * phi_dr + self.alpha * self.Lm * i_qs
        dx5 = self.mu / self.J * (phi_dr * i_qs - phi_qr * i_ds) - self.tl_star / self.J

        # fx = [-gamma*i_ds + omega_1*i_qs + beta*(alpha*phi_dr + omega*phi_qr);
        # -gamma*i_qs - omega_1*i_ds + beta*(alpha*phi_qr - omega*phi_dr);
        # -alpha*phi_dr + (omega_1 - omega)*phi_qr + alpha*Lm*i_ds;
        # -alpha*phi_qr - (omega_1 - omega)*phi_dr + alpha*Lm*i_qs;
        # mu/J*(phi_dr*i_qs - phi_qr*i_ds) - Tl/J];

        return np.array([dx1, dx2, dx3, dx4, dx5], dtype=np.float32)


if __name__ == '__main__':
    env = InductionMachineEnv()
    print(env.reset(options={
        'init_state': np.array([env.id_ref, env.iq_ref, env.phi_star, 0., env.omega_star])
    }))
    states = []
    rewards = []

    done = False
    while not done:
        state, reward, terminated, truncated, _ = env.step(env.equilibirum_action)
        rewards.append([reward])
        states.append(state)
        done = terminated or truncated

    states = np.array(states)
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.show()
