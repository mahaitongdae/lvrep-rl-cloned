import torch
import numpy as np
import gymnasium
from differentiable_dynamics import Pendulum3D


class NystromFeatureExtractor(object):
    def __init__(self, env, sample_dim=512):
        # env = gymnasium.make('Pendulum-v1')
        self.rollout_dynamics = None
        self.sigma = None
        self.s_high = np.clip(env.observation_space.high, -10., 10.).tolist()
        self.s_low = np.clip(env.observation_space.low, -10., 10.).tolist()  # in case of inf observation space
        self.s_dim = env.observation_space.shape if len(env.observation_space.shape) > 1 else \
            env.observation_space.shape[0]
        self.a_high = np.clip(env.action_space.high, -10., 10.).tolist()
        self.a_low = np.clip(env.action_space.low, -10., 10.).tolist()
        self.a_dim = env.action_space.shape if len(env.action_space.shape) > 1 else env.action_space.shape[0]
        self.env = env
        self.sample_dim = sample_dim
        # self.sample_and_decompose()

    def set_dynamics(self, dynamics, sigma=1.):
        self.rollout_dynamics = dynamics
        self.sigma = sigma
        if self.sigma > 0.0:
            self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2. * self.sigma ** 2))
        else:
            self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2.))

    def random_sample_sprime_fsa(self):
        nystrom_samples = np.random.uniform(self.s_low, self.s_high, size=(self.sample_dim, self.s_dim))
        action_sample = np.random.uniform(self.a_low, self.a_high, size=(self.sample_dim, self.a_dim))
        fsa = self.rollout_dynamics.rollout(torch.from_numpy(nystrom_samples), torch.from_numpy(action_sample))
        fsa = fsa.clone().detach().numpy()
        sigma = self.sigma if self.sigma != 0. else 1.
        noise = np.random.normal(scale=sigma * 0.05, size=fsa.shape)  # TODO: the scale of noise
        sprime = fsa + noise
        return sprime, fsa

    def sample_and_decompose(self, sample='random', decompose='numpy'):
        """get nystrom feature from samples

        Args:
            sample (str, optional): Sampling methods. Defaults to 'random'.
            decompose (str, optional): decomposition backend. Defaults to 'numpy'. 'torch' or 'numpy'.
        """
        if sample == 'random':
            sprime, fsa = self.random_sample_sprime_fsa()
        else:
            raise NotImplementedError
        # K_m1 = self.get_kernel_matrix_asymmetric(sprime, fsa)
        K_m1 = self.kernel_matrix_torch(sprime, fsa)
        U, singular_vals, VT = np.linalg.svd(K_m1)
        eig_val = np.clip(singular_vals, 1e-8, np.inf)
        self.eig_val = torch.from_numpy(eig_val).float()
        self.S = torch.from_numpy(U).float()
        self.nystrom_sample = torch.from_numpy(fsa).float()

    @staticmethod
    def kernel_matrix_torch(x1, x2):
        if isinstance(x1, np.ndarray):
            x1 = torch.from_numpy(x1)
        if isinstance(x2, np.ndarray):
            x2 = torch.from_numpy(x2)
        # dx = x1.unsqueeze(0) - x2.unsqueeze(1)
        dx = x1.unsqueeze(1) - x2.unsqueeze(0)  # will return the kernel matrix of k(x1, x2) with symmetric kernel.
        K_x1 = torch.exp(-torch.linalg.norm(dx, axis=2) ** 2 / 2).float()
        return K_x1

    @staticmethod
    def kernel_matrix_numpy(x1, x2):
        dx2 = np.expand_dims(x1, axis=1) - np.expand_dims(x2,
                                                          axis=0)  # will return the kernel matrix of k(x1, x2) with symmetric kernel.
        K_x2 = np.exp(-np.linalg.norm(dx2, axis=2) ** 2 / 2)
        return K_x2

    def get_nystrom_feature(self, input):
        # x1 = self.nystrom_sample.unsqueeze(0) - input.unsqueeze(1)
        # K_x1 = torch.exp(-torch.linalg.norm(x1, axis=2) ** 2 / 2).float()
        K_x = self.kernel_matrix_torch(input, self.nystrom_sample).float()
        phi = (K_x @ (self.S)) @ torch.diag((self.eig_val + 1e-8) ** (-0.5))
        return phi

    def get_kernel_matrix_asymmetric(self, sprime, fsa):
        m = sprime.shape[0]
        n = fsa.shape[0]
        K_m = np.empty((m, n))
        for i in np.arange(m):
            for j in np.arange(n):
                K_m[i, j] = self.kernel(sprime[i, :] - fsa[j, :])
        return K_m


def pendulum_3d(obs, action, g=10.0, m=1., l=1., max_a=2., max_speed=8., dt=0.05):
    th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
    thdot = obs[:, 2]
    action = torch.reshape(action, (action.shape[0],))
    u = torch.clip(action, -max_a, max_a)
    newthdot = thdot + (3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = torch.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    new_states = torch.empty((obs.shape[0], 3))
    new_states[:, 0] = torch.cos(newth)
    new_states[:, 1] = torch.sin(newth)
    new_states[:, 2] = newthdot
    return new_states


def test_numpy_torch_kernel_matrix():
    pendulum_3d = Pendulum3D()
    env = gymnasium.make('Pendulum-v1')
    feature_extractor = NystromFeatureExtractor(env)
    feature_extractor.set_dynamics(pendulum_3d)
    sprime, fsa = feature_extractor.random_sample_sprime_fsa()
    K_torch = feature_extractor.kernel_matrix_torch(sprime, fsa)
    K_numpy = feature_extractor.kernel_matrix_numpy(sprime, fsa)
    return torch.equal(torch.from_numpy(K_numpy).float(), K_torch)


def test_feature_extractor():
    pendulum_3d = Pendulum3D(g=120., m=10.,)
    env = gymnasium.make('Pendulum-v1')
    feature_extractor = NystromFeatureExtractor(env)
    feature_extractor.set_dynamics(pendulum_3d)
    feature_extractor.sample_and_decompose()
    phi = feature_extractor.get_nystrom_feature(
        torch.reshape(torch.from_numpy(env.observation_space.sample()), [1, -1]))
    return phi


if __name__ == '__main__':
    print(test_feature_extractor())
