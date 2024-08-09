from repr_control.agent.rfsac.rfsac_agent import CustomModelRFSACAgent

import torch
from torch import nn
import numpy as np
import copy
from repr_control.utils.util import unpack_batch
import torch.nn.functional as F

class TwoDimensionalGaussianCritic(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros((2, ), device=device))
        self.std_var = torch.nn.Parameter(torch.eye(2, device=device))

    def forward(self, inputs: torch.Tensor):
        std = self.std_var @ (self.std_var.T)
        std_inv = torch.linalg.inv(std)
        bias = inputs - self.mean.unsqueeze(0)
        exponent = bias.unsqueeze(1) @ std_inv @ bias.unsqueeze(2)
        exponent.squeeze_()
        density = 1 / (2 * np.pi * (torch.linalg.det(std) ** 0.5)) * torch.exp(
            -0.5 * exponent)
        return density

class DensityEvaluationAgent(CustomModelRFSACAgent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 dynamics_fn,
                 rewards_fn,
                 lr=3e-4,
                 discount=0.99,
                 target_update_period=2,
                 tau=0.005,
                 alpha=0.1,
                 auto_entropy_tuning=True,
                 hidden_dim=256,
                 sigma=0.0,
                 rf_num=256,
                 learn_rf=False,
                 use_nystrom=False,
                 replay_buffer=None,
                 device = 'cpu',
                 **kwargs
                 ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            dynamics_fn=dynamics_fn,
            rewards_fn=rewards_fn,
            action_range=action_range,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            sigma=sigma,
            rf_num=rf_num,
            learn_rf=learn_rf,
            use_nystrom=use_nystrom,
            replay_buffer=replay_buffer,
            device=device,
            **kwargs)

        self.density = TwoDimensionalGaussianCritic(device=torch.device(device))
        self.density_optimizer = torch.optim.Adam(self.density.parameters(), lr, betas=[0.9, 0.999])
        self.density_target = copy.deepcopy(self.density)
        self.init_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros(2, device=device),
                                                                   covariance_matrix=torch.tensor([[1.0,0.5],
                                                                                                   [0.5, 1.0]], device=device))
        # self.actor = None


    def density_critic_step(self, batch, alg='td'):
        """
        Critic update step
        alg: td or mle
        """
        # state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
        state, action, next_state, _, done = unpack_batch(batch)

        if alg == 'td':

            with torch.no_grad():
                density1 = torch.exp(self.init_dist.log_prob(state))
                # if self.density.log_prob:
                #     density1, density2 = torch.exp(density1), torch.exp(density2)
                # initial_density_state = self.get_initial_density(state).unsqueeze(1)
                action_dist = self.actor(state)
                action_prob = torch.ones_like(density1) # fixed det policy at 0
                # action_prob = torch.exp(action_log_prob)
                target_next_density = torch.multiply(density1, action_prob)

            next_density1 = self.density(next_state)
            density1_loss = F.mse_loss(next_density1, target_next_density)
            d_loss = density1_loss#  + density2_loss

            self.density_optimizer.zero_grad()
            d_loss.backward()
            self.density_optimizer.step()

            info = {
                'd1_loss': density1_loss.item(),
                # 'd2_loss': density2_loss.item(),
                'd1': density1.mean().item(),
                # 'd2': density2.mean().item(),
                'layer_norm_weights_norm': self.critic.norm.weight.norm(),
            }

            dist = {
                'density_td_error': (next_density1 - target_next_density).cpu().detach().clone().numpy(),
                'density': density1.cpu().detach().clone().numpy()
            }

            info.update({'density_critic_dist': dist})

            return info

        elif alg == 'mle':

            log_likeligood = torch.log(self.density(next_state))

            d_loss = -1 * log_likeligood.mean()

            self.density_optimizer.zero_grad()
            d_loss.backward()
            self.density_optimizer.step()

            info = {
                'd1_loss': d_loss.item(),
                # 'd2_loss': density2_loss.item(),
                # 'd1': density1.mean().item(),
                # 'd2': density2.mean().item(),
                # 'layer_norm_weights_norm': self.critic.norm.weight.norm(),
            }

            return info

        elif alg == 'grad_td':

            log_density = self.init_dist.log_prob(state)



    def batch_train(self, batch):
        """
                One train step
                """
        self.steps += 1

        # Acritic step
        critic_info = self.density_critic_step(batch)

        # Actor and alpha step
        # actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        # self.update_target()

        return {
            **critic_info,
            # **actor_info,
        }