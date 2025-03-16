import torch
import numpy as np
state_dim = 5                       # state dimension
action_dim = 2                      # action dimension
state_range = [
    [
    -1000.,
    -1000.,
    -1000.,
    -1000.,
    -1000.,
    ],
    [
    1000.,
    1000.,
    1000.,
    1000.,
    1000.,
    ],
]           # low and high. We set bound on the state to ensure stable training.


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
Q = [1., 0.1, 1000., ]
R = [0.001, 0.001]

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

action_range = [[-1, -1], [1, 1]]          # low and high
max_step = 1000                      # maximum rollout steps per episode
sigma = 0.0                          # noise standard deviation.
env_name = 'Parking'
assert len(action_range[0]) == len(action_range[1]) == action_dim
terminal_threshold = torch.tensor([[0.3, 0.3, np.pi / 24, np.pi / 24, 0.1, np.inf]], device=torch.device('cuda'))

def dynamics(state, action):
    l = 4.9276  # tractor length
    R = 8.5349  # turning radius
    d1 = 15.8496  # trailer length
    delta_max = np.pi / 6
    dt = 0.05

    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc = action[:, 0]
    delta_rate = action[:, 1] * delta_max / 4
    normalized_steer = torch.tan(delta) * R / l

    ds = torch.vstack([
        v * torch.cos(th0),
        v * torch.sin(th0),
        v * normalized_steer / R,
        -1 * v * (d1 * normalized_steer + torch.sin(dth) * R) / (R * d1),
        acc,
        delta_rate
    ]).T

    stp1 = state + ds * dt
    return stp1

def target_reached(state):

    reached = torch.all(torch.abs(state) < terminal_threshold, dim=1)
    return reached

def rewards(state, action, terminal = False):
    x, y, th0, dth, v, delta = torch.unbind(state, dim=1)
    acc, delta_rate = torch.unbind(action, dim=1)
    # if not terminal:
    reward = -1e-1 * (x ** 2 + y ** 2
                      + 10 * th0 ** 2
                      + 10 * dth ** 2
                      + v ** 2
                      + delta ** 2
                      + acc ** 2
                      + delta_rate ** 2)
    reached = target_reached(state)
    mod_reward = torch.where(reached, 100 * torch.ones_like(reward), reward)
    return mod_reward

def initial_distribution(batch_size):

    high = np.array(
            [
                10,
                3,
                np.pi / 6,
                np.pi / 6,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    low = np.array(
        [0.0,
            0.0,
            -np.pi / 12,
            -np.pi / 12,
            0.0,
            0.0
            ]
    )

    high = np.array(
        [
            8.0,
            4.0,
            np.pi / 12,
            np.pi / 12,
            0.0,
            0.0
        ]
    )
    # self.state = self.np_random.uniform(low=-1 * high, high=high)
    state = np.random.uniform(low, high, size=(batch_size, len(high)))
    state = np.clip(state, -high, high)
    return torch.from_numpy(state)

def get_done(state):
    done = torch.all(torch.abs(state) < terminal_threshold, dim=1)
    return done


def test_custom_env():
    # from repr_control.scripts.define_problem import dynamics, rewards, initial_distribution, state_range, action_range, sigma
    from repr_control.envs.custom_env import CustomVecEnv
    env = CustomVecEnv(dynamics, rewards, initial_distribution, state_range, action_range, sigma)

    print(env.reset())
    for i in range(10):
        print(env.step(env.sample_action()))