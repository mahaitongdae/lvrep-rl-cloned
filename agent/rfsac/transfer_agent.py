from agent.rfsac.rfsac_agent import nystromVCritic
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleTransferAgent(object):
    
    def __init__(self, **kwargs):
        self.s_low = kwargs.get('obs_space_low')
        self.s_high = kwargs.get('obs_space_high')
        self.s_dim = kwargs.get('obs_space_dim')
        self.a_low = -1.0
        self.a_high = 1.0
        self.s_dim = self.s_dim[0] if (not isinstance(self.s_dim, int)) else self.s_dim
        self.feature_dim = kwargs.get('random_feature_dim')
        self.sample_dim = kwargs.get('nystrom_sample_dim')
        self.sigma = kwargs.get('sigma')
        self.dynamics_type = kwargs.get('dynamics_type')
        self.sin_input = kwargs.get('dynamics_parameters').get('sin_input')
        self.dynamics_parameters = kwargs.get('dynamics_parameters')
        self.kwargs = kwargs

    def set_env(self, env):
        self.env = env
        self.g = torch.tensor(self.env.g, requires_grad=True)
        self.m = torch.tensor(self.env.m, requires_grad=True)
        self.l = torch.tensor(self.env.l, requires_grad=True)

    def f_star_3d(self, states, action, max_a=2., max_speed=8., dt=0.05):
        th = torch.atan2(states[:, 1], states[:, 0])  # 1 is sin, 0 is cosine
        thdot = states[:, 2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action, -max_a, max_a)
        newthdot = thdot + (3. * self.g / (2 * self.l) * torch.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt
        return torch.vstack([torch.cos(newth), torch.sin(newth), newthdot]).T

    def differentiable_nystrom(self):

        np.random.seed(self.kwargs.get('seed'))
        # create nystrom feats
        nystrom_samples_s = np.random.uniform(self.s_low, self.s_high, size=(self.sample_dim, self.s_dim))
        nystrom_samples_s = torch.from_numpy(nystrom_samples_s)
        nystrom_samples_a = np.random.uniform(self.a_low, self.a_high, size=(self.sample_dim))
        nystrom_samples_a = torch.from_numpy(nystrom_samples_a)
        np.random.seed(self.kwargs.get('seed') + 1)
        nystrom_samples_s_prime = torch.from_numpy(np.random.uniform(self.s_low, self.s_high, size=(self.sample_dim, self.s_dim)))

        if self.sigma > 0.0:
            self.kernel = lambda z: torch.exp(-torch.linalg.norm(z) ** 2 / (2. * self.sigma ** 2))
        else:
            self.kernel = lambda z: torch.exp(-torch.linalg.norm(z) ** 2 / (2.))
        K_m1 = self.make_K(nystrom_samples_s, nystrom_samples_a, nystrom_samples_s_prime)
        print('start eig')

        [eig_vals1, S1] = np.linalg.eig(K_m1)  # numpy linalg eig doesn't produce negative eigenvalues... (unlike torch)

        # truncate top k eigens
        argsort = np.argsort(eig_vals1)[::-1]
        eig_vals1 = eig_vals1[argsort]
        S1 = S1[:, argsort]
        eig_vals1 = np.clip(eig_vals1, 1e-8, np.inf)[:self.feature_dim]
        self.eig_vals1 = torch.from_numpy(eig_vals1).float().to(device)
        self.S1 = torch.from_numpy(S1[:, :self.feature_dim]).float().to(device)
        self.nystrom_samples1 = torch.from_numpy(self.nystrom_samples1).to(device)

    def kernel(self, s, a, s_prime):
        if len(s.shape) == 1:
            s = torch.reshape(s, [1, self.s_dim])
        if len(a.shape) == 0:
            a = torch.reshape(a, [1,])
        f_sa = self.f_star_3d(s, a)


    def make_K(self, sample_s, sample_a, sample_s_prime):
        print('start cal K')
        m,d = sample_s_prime.shape
        sample_f_sa = self.f_star_3d(sample_s, sample_a)
        K_m = np.empty((m,m))
        for i in np.arange(m):
            for j in np.arange(m):
                K_m[i,j] = self.kernel(sample_f_sa[i,:] - samples[j,:])
        return K_m
