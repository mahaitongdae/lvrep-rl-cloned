import torch
import torch.nn.functional as F
from repr_control.utils.buffer import Batch
from repr_control.agent.sac.sac_agent import ModelBasedSACAgent
from repr_control.agent.actor import DeterministicActor
from repr_control.agent.critic import DoubleQCritic
from repr_control.agent.actor import DeterministicActor, DeterministicQPActor
import numpy as np
from repr_control.utils import util

class DPGAgent(object):
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
		self.actor = DeterministicActor(
			obs_dim=state_dim,
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=hidden_depth,
		).to(self.device)
		# self.log_alpha = torch.tensor(np.log(alpha)).float().to(self.device)
		# self.log_alpha.requires_grad = True
		# self.target_entropy = -action_dim

		# optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])
		critic_lr = kwargs['critic_lr'] if 'critic_lr' in kwargs.keys() else lr
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
												 lr=critic_lr,
												 betas=[0.9, 0.999])

	# @property
	# def alpha(self):
	# 	return self.log_alpha.exp()

	def select_action(self, state, explore=False):
		if isinstance(state, list):
			state = np.array(state)
		assert len(state.shape) == 1
		state = torch.from_numpy(state).to(self.device)
		state = state.unsqueeze(0)
		action = self.actor(state)
		# action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])

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

		next_action = self.actor(next_obs)
		 # = dist.rsample()
		# log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
		target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
		target_V = torch.min(target_Q1,
							 target_Q2)#  - self.alpha.detach() * log_prob
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

		action = self.actor(obs)
		# action = dist.rsample()
		# log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_Q1, actor_Q2 = self.critic(obs, action)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (- actor_Q).mean() # self.alpha.detach() * log_prob

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		info = {'actor_loss': actor_loss.item()}

		# if self.learnable_temperature:
		# 	self.log_alpha_optimizer.zero_grad()
		# 	alpha_loss = (self.alpha *
		# 				  (-log_prob - self.target_entropy).detach()).mean()
		# 	alpha_loss.backward()
		# 	self.log_alpha_optimizer.step()
		#
		# 	info['alpha_loss'] = alpha_loss
		# 	info['alpha'] = self.alpha

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

class ModelBasedDPGAgent(ModelBasedSACAgent):

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
		super().__init__(state_dim,
				 action_dim,
				 action_range,
				 dynamics,
				 rewards,
				 initial_distribution,
				 horizon=horizon,
				 lr=lr,
				 discount=discount,
				 target_update_period=target_update_period,
				 tau=tau, alpha=alpha,
				 auto_entropy_tuning=auto_entropy_tuning,
				 hidden_dim=hidden_dim,
				 hidden_depth=hidden_depth,
				 device=device,
				 **kwargs)
		self.actor = DeterministicActor(state_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999))
		self.actor_supervised_optimizer = torch.optim.Adam(self.actor.parameters(),
														   lr=1e-3,
														   betas=[0.9, 0.999])
		self.cost_to_go = util.mlp(state_dim, hidden_dim, 1, hidden_depth).to(self.device)
		self.cost_to_go.apply(util.weight_init)
		self.cost_to_go_optimizer = torch.optim.Adam(self.cost_to_go.parameters(), lr=3e-4, betas=[0.9,0.999])

	def update_actor_and_alpha(self, batch):
		obs = batch.state
		log_probs = []
		rewards = torch.zeros([obs.shape[0]]).to(self.device)
		for i in range(self.horizon):
			action = self.actor(obs)
			# action = dist.rsample()
			# log_prob = dist.log_prob(action).sum(-1, keepdim=True)
			obs = self.dynamics(obs, action)
			if i == self.horizon - 1:
				rewards += self.rewards(obs, action, terminal=True)
			else:
				rewards += self.rewards(obs, action, terminal=False)
			# log_probs.append(log_prob)
		final_reward = self.rewards(obs, action, terminal=True)
		actor_loss = -1 * rewards.mean()
		actor_loss_value = actor_loss.clone().detach()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		v = self.cost_to_go(obs)
		critic_loss = ((v - actor_loss_value) ** 2).mean()
		self.cost_to_go_optimizer.zero_grad()
		critic_loss.backward(inputs = list(self.cost_to_go.parameters()))
		self.cost_to_go_optimizer.step()

		info = {'actor_loss': actor_loss.item(),
				'terminal_cost': final_reward.mean().item(),
				'critic_loss': critic_loss.item()}

		return info

	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		state = self.initial_dist(batch_size).float().to(self.device)
		batch = Batch(state=state,
					  action=None,
					  next_state=None,
					  reward=None,
					  done=None, )

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			# **critic_info,
			**actor_info,
		}

	def supervised_from_mpc(self, batch):

		obs, action = batch
		if obs.device == torch.device('cpu'):
			obs = obs.float().to(self.device)
			action = action.float().to(self.device)

		if obs.shape[1] == 7:
			obs = obs[:, :-1]
		output = self.actor(obs)
		loss = F.mse_loss(output, action)

		# optimize the actor
		self.actor_supervised_optimizer.zero_grad()
		loss.backward()
		self.actor_supervised_optimizer.step()

		info = {'supervised_loss': loss.item()}

		return info


	def supervised_train(self, batch,):
		actor_info = self.supervised_from_mpc(batch)
		# critic_info = self.critic_step(batch, su)

		return actor_info

class ModelBasedQPDPGAgent(ModelBasedDPGAgent):

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
		super().__init__(state_dim,
				 action_dim,
				 action_range,
				 dynamics,
				 rewards,
				 initial_distribution,
				 horizon=horizon,
				 lr=lr,
				 discount=discount,
				 target_update_period=target_update_period,
				 tau=tau, alpha=alpha,
				 auto_entropy_tuning=auto_entropy_tuning,
				 hidden_dim=hidden_dim,
				 hidden_depth=hidden_depth,
				 device=device,
				 **kwargs)

		self.actor = DeterministicQPActor(state_dim, action_dim,
										  hidden_dim, hidden_depth,
										  self.device, kwargs.get('n_qp', 12),
										  kwargs.get('m_qp', 48))
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])
		self.action_dim = action_dim
		self.actor_supervised_optimizer = torch.optim.Adam(self.actor.parameters(),
														   lr=1e-3,
														   betas=[0.9, 0.999])

	def update_actor_and_alpha(self, batch):
		obs = batch.state
		log_probs = []
		rewards = torch.zeros([obs.shape[0]]).to(self.device)
		for i in range(self.horizon):
			action = self.actor(obs)
			obs = self.dynamics(obs, action)
			if i == self.horizon - 1:
				rewards += self.rewards(obs, action, terminal=True)
			else:
				rewards += self.rewards(obs, action, terminal=False)
		final_reward = self.rewards(obs, action, terminal=True)
		actor_loss = -1 * rewards.mean()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		info = {'actor_loss': actor_loss.item(),
				'terminal_cost': final_reward.mean().item()}

		return info


class ModelBasedDPGAgentTerminalConstraints(ModelBasedDPGAgent):

	def __init__(self, state_dim, 
					action_dim, 
					action_range, 
					dynamics, 
					rewards, 
					initial_distribution, 
					terminal_constraints,
					action_noise,
					lr_schedule=False,
					statewise_weights=True,
					horizon=250, 
					lr=0.0003, 
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
				   dynamics, 
				   rewards, 
				   initial_distribution, 
				   horizon, 
				   lr, 
				   discount, 
				   target_update_period, tau, alpha, auto_entropy_tuning, hidden_dim, hidden_depth, device, **kwargs)
		
		self.actor = DeterministicActor(state_dim+state_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
												lr=lr,
												betas=[0.9, 0.999])
		self.actor_supervised_optimizer = torch.optim.Adam(self.actor.parameters(),
														   lr=1e-3,
														   betas=[0.9, 0.999])
		self.action_noise_std = action_noise

		self.terminal_constraints = terminal_constraints
		if not statewise_weights:
			self.terminal_constraint_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).unsqueeze(dim=0).to(self.device).requires_grad_()
			self.terminal_constraint_weights_optimizer = torch.optim.Adam(params=[self.terminal_constraint_weights], lr=1e-2)
		else:
			self.terminal_constraint_weights = util.mlp(state_dim, hidden_dim, 4, hidden_depth, output_mod=torch.nn.Softplus()).to(self.device)
			self.terminal_constraint_weights_optimizer = torch.optim.Adam(params=self.terminal_constraint_weights.parameters(), lr=0.3 * lr)
		self.statewise_weights = statewise_weights
		self.lr_schedule = lr_schedule
		if lr_schedule:
			from torch.optim.lr_scheduler import LinearLR
			self.lr_scheduler_actor = LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=int(kwargs['max_timesteps'] / 2))
			self.lr_scheduler_weights = LinearLR(self.terminal_constraint_weights_optimizer, start_factor=1.0, end_factor=0.1, total_iters=int(kwargs['max_timesteps'] / 10))

	
	def update_actor_and_alpha(self, batch):
		obs = batch.state
		init_state = batch.state
		if self.statewise_weights:
			weights = self.terminal_constraint_weights(obs)
		rewards = torch.zeros([obs.shape[0]]).to(self.device)
		for i in range(self.horizon):
			action = self.actor(torch.hstack([obs, init_state]))
			if self.action_noise_std > 0:
				noise = self.action_noise_std * torch.randn_like(action)
				action = torch.clamp(action + noise, min=-1, max=1)
			obs = self.dynamics(obs, action)
			rewards += self.rewards(obs, action)
		terminal_constraint = self.terminal_constraints(obs)
		if self.statewise_weights:
			weighted_terminal_constraint = (weights * terminal_constraint).sum(dim=1)
		else:
			weighted_terminal_constraint = (self.terminal_constraint_weights * terminal_constraint).sum(dim=1)
		actor_loss = (-1 * rewards +  weighted_terminal_constraint).mean()
		weights_loss = (-1 * weighted_terminal_constraint).mean()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		self.terminal_constraint_weights_optimizer.zero_grad()
		actor_loss.backward(inputs=list(self.actor.parameters()), retain_graph=True)
		if self.steps % 5 == 0:
			if not self.statewise_weights:
				weights_loss.backward(inputs=[self.terminal_constraint_weights])
			else:
				weights_loss.backward(inputs=list(self.terminal_constraint_weights.parameters()))
		self.actor_optimizer.step()
		self.lr_scheduler_actor.step()
		if self.steps % 5 == 0:
			self.terminal_constraint_weights_optimizer.step()
			if not self.statewise_weights:
				with torch.no_grad():
					self.terminal_constraint_weights.clamp_(min=0.0)
			self.lr_scheduler_weights.step()

		# self.terminal_constraint_weights = torch.clip(self.terminal_constraint_weights, 
		# 										min = torch.zeros_like(self.terminal_constraint_weights)).detach_().requires_grad_()

		# v = self.cost_to_go(obs)
		# critic_loss = ((v - actor_loss_value) ** 2).mean()
		# self.cost_to_go_optimizer.zero_grad()
		# critic_loss.backward(inputs = list(self.cost_to_go.parameters()))
		# self.cost_to_go_optimizer.step()

		info = {'actor_loss': actor_loss.item(),
		  		'weights_loss': weights_loss.item(),
		  		'average_cstr_1x' : terminal_constraint[:, 0].mean().item(),
				'average_cstr_2y' : terminal_constraint[:, 1].mean().item(),
				'average_cstr_3th' : terminal_constraint[:, 2].mean().item(),
				'average_cstr_4dth' : terminal_constraint[:, 3].mean().item(),
				
				'avg_reward': rewards.mean().item(),
				'terminal_cost': weighted_terminal_constraint.mean().item(),
				# 'critic_loss': critic_loss.item(),
				}
		
		if not self.statewise_weights:
			info.update({
				'weights_1x': self.terminal_constraint_weights[:, 0].item(),
				'weights_2y': self.terminal_constraint_weights[:, 1].item(),
				'weights_3th': self.terminal_constraint_weights[:, 2].item(),
				'weights_4th0': self.terminal_constraint_weights[:, 3].item(),
			})

		return info
