import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from repr_control.utils import util
from repr_control.utils.buffer import Batch
from repr_control.agent.critic import DoubleQCritic
from repr_control.agent.actor import DiagGaussianActor, DeterministicActor, StochasticActorFromDetStructureWrapper
from repr_control.networks.qp.qp_unrolled_network import QPUnrolledNetwork, mlp_builder


class SACAgent(object):
	"""
	DDPG Agent
	"""

	def __init__(
			self,
			state_dim,
			action_dim,
			action_range,
			lr=3e-4,
			discount=0.99,
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=1024,
			hidden_depth=2,
			device='cpu',
			**kwargs
	):

		self.steps = 0

		self.device = torch.device(device)
		self.action_range = action_range
		self.discount = discount
		self.tau = tau
		self.target_update_period = target_update_period
		self.learnable_temperature = auto_entropy_tuning

		# functions
		self.critic = DoubleQCritic(
			obs_dim=state_dim,
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=hidden_depth,
		).to(self.device)
		self.critic_target = DoubleQCritic(
			obs_dim=state_dim,
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=hidden_depth,
		).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.actor = DiagGaussianActor(
			obs_dim=state_dim,
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=hidden_depth,
			log_std_bounds=[-5., 2.],
		).to(self.device)
		self.log_alpha = torch.tensor(np.log(alpha)).float().to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -action_dim

		# optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])
		critic_lr = kwargs['critic_lr'] if 'critic_lr' in kwargs.keys() else lr
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
												 lr=critic_lr,
												 betas=[0.9, 0.999])

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
													lr=lr,
													betas=[0.9, 0.999])

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def select_action(self, state, explore=False):
		if isinstance(state, list):
			state = np.array(state)
		assert len(state.shape) == 1
		state = torch.from_numpy(state).to(self.device)
		state = state.unsqueeze(0)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])

	def batch_select_action(self, state, explore=False):
		assert isinstance(state, torch.Tensor)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		return action


	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def critic_step(self, batch):
		"""
		Critic update step
		"""
		obs, action, next_obs, reward, done = util.unpack_batch(batch)
		not_done = 1. - done

		dist = self.actor(next_obs)
		next_action = dist.rsample()
		log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
		target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
		target_V = torch.min(target_Q1,
							 target_Q2) - self.alpha.detach() * log_prob
		target_Q = reward + (not_done * self.discount * target_V)
		target_Q = target_Q.detach()

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
			current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return {
			'q_loss': critic_loss.item(),
			'q1': current_Q1.mean().item(),
			'q2': current_Q1.mean().item()
		}

	def update_actor_and_alpha(self, batch):
		obs = batch.state

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_Q1, actor_Q2 = self.critic(obs, action)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		info = {'actor_loss': actor_loss.item()}

		if self.learnable_temperature:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha *
						  (-log_prob - self.target_entropy).detach()).mean()
			alpha_loss.backward()
			self.log_alpha_optimizer.step()

			info['alpha_loss'] = alpha_loss
			info['alpha'] = self.alpha

		return info

	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)
		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			**critic_info,
			**actor_info,
		}

	def batch_train(self, batch):

		"""
				One train step
				"""
		self.steps += 1

		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			**critic_info,
			**actor_info,
		}

class QPSACAgent(SACAgent):

	def __init__(self,
			state_dim,
			action_dim,
			action_range,
			lr=3e-4,
			discount=0.99,
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=1024,
			hidden_depth=2,
			device='cpu',
			**kwargs):
		super().__init__(state_dim,
			action_dim,
			action_range,
			lr=lr,
			discount=discount,
			target_update_period=target_update_period,
			tau=tau,
			alpha=alpha,
			auto_entropy_tuning=auto_entropy_tuning,
			hidden_dim=hidden_dim,
			hidden_depth=hidden_depth,
			device=device,
			**kwargs)
		from repr_control.networks.qp import qp_default_args
		qp_default_args.update({
						 'device': device, # device, input_size, n_qp, m_qp, qp_iter, mlp_builder,
                         'input_size': state_dim,
                         'n_qp': kwargs.get('n_qp', 8),             #  P is n by n
                         'm_qp': kwargs.get('m_qp', 48),            # m is constraint nums
                         'qp_iter' : 10,
                         'mlp_builder': mlp_builder
                         })
		qp_policy = QPUnrolledNetwork(**qp_default_args)
		self.actor = StochasticActorFromDetStructureWrapper(state_dim,
															action_dim,
															hidden_dim,
															hidden_depth,
															[-5., 2.],
															qp_policy).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])


class ModelBasedSACAgent(SACAgent):

	def __init__(self, state_dim,
				 action_dim,
				 action_range,
				 dynamics,
				 rewards,
				 initial_distribution,
				 horizon=250,
				 lr=0.0003,
				 discount=0.99,
				 target_update_period=2,
				 tau=0.005, alpha=0.1,
				 auto_entropy_tuning=True,
				 hidden_dim=1024,
				 hidden_depth=2,
				 device='cpu',
				 **kwargs):
		super().__init__(state_dim, action_dim, action_range, lr, discount, target_update_period, tau, alpha,
						 auto_entropy_tuning, hidden_dim, hidden_depth, device, **kwargs)
		self.horizon = horizon
		self.dynamics = dynamics
		self.rewards = rewards
		self.initial_dist = initial_distribution

	def update_actor_and_alpha(self, batch):
		obs = batch.state
		log_probs = []
		rewards = torch.zeros([obs.shape[0]]).to(self.device)
		for i in range(self.horizon):
			dist = self.actor(obs)
			action = dist.rsample()
			log_prob = dist.log_prob(action).sum(-1, keepdim=True)
			obs = self.dynamics(obs, action)
			if i == self.horizon - 1:
				rewards += self.rewards(obs, action, terminal=True)
			else:
				rewards += self.rewards(obs, action, terminal=False)
			log_probs.append(log_prob)
		final_reward = self.rewards(obs, action, terminal=True)
		actor_loss = -1 * rewards.mean()
		log_prob_all = torch.hstack(log_probs)

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		info = {'actor_loss': actor_loss.item(),
				'terminal_cost': final_reward.mean().item()}

		if self.learnable_temperature:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha *
						  (-log_prob_all - self.target_entropy).detach()).mean()
			alpha_loss.backward()
			self.log_alpha_optimizer.step()

			info['alpha_loss'] = alpha_loss.item()
			info['alpha'] = self.alpha.item()

		return info

	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		state = torch.from_numpy(self.initial_dist(batch_size)).float().to(self.device)
		batch = Batch(state=state,
					  action=None,
					  next_state=None,
					  reward=None,
					  done=None, )
		# Acritic step
		# critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			# **critic_info,
			**actor_info,
		}


def test_fh_agent_biagt():
	from repr_control.envs.models.articulate_model_fh import dynamics, reward, initial_distribution
	agent = ModelBasedSACAgent(7, 2, [[-1, -1], [1, 1]], dynamics, reward, initial_distribution)
	agent.train(None, batch_size=256)


if __name__ == '__main__':
	test_fh_agent_biagt()

# test_fh_agent_biagt()
