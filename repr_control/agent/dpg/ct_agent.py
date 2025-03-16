from repr_control.agent.dpg.dpg_agent import DPGAgent
import torch
import torch.nn.functional as F
from repr_control.utils.buffer import Batch
from repr_control.agent.sac.sac_agent import ModelBasedSACAgent
from repr_control.agent.actor import DeterministicActor
from repr_control.agent.critic import DoubleQCritic, DoubleDerivativeQCritic
from repr_control.agent.actor import DeterministicActor, DeterministicQPActor
import numpy as np
from repr_control.utils import util


class ContinuousTimeDPGAgent(DPGAgent):

    def __init__(self, state_dim,
                 action_dim,
                 action_range,
                 dynamics,
                 rewards,
                 initial_distribution,
                 horizon = 250,
                 lr = 0.0003,
                 discount = 0.99,
                 target_update_period = 2,
                 tau = 0.005, alpha = 0.1,
                 auto_entropy_tuning = True,
                 hidden_dim = 1024,
                 hidden_depth = 2,
                 device = 'cpu',
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
        self.actor_supervised_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                           lr=1e-3,
                                                           betas=[0.9, 0.999])
        self.critic = DoubleDerivativeQCritic(state_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleDerivativeQCritic(state_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.dynamics = dynamics
        self.rewards = rewards
        self.initial_dist = initial_distribution

    def update_critic(self, batch):
        obs, action, reward = batch.state, batch.action, batch.reward
        partial1, partial2 = self.critic_target(obs) # [batch_size, state]
        fxu = self.dynamics(obs, action) # [batch_size, state]
        loss1 = reward + torch.sum(partial1 * fxu, dim=1, keepdim=False)
        loss2 = reward + torch.sum(partial2 * fxu, dim=1, keepdim=False)
        loss = loss1 ** 2 + loss2 ** 2
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {
            "q_loss": loss.item()
        }


    def update_actor_and_alpha(self, batch):
        obs = batch.state
        log_probs = []
        rewards = torch.zeros([obs.shape[0]]).to(self.device)
        action = self.actor(obs)
        partial1, partial2 = self.critic_target(obs)
        fxu = self.dynamics(obs, action)  # [batch_size, state]
        loss = torch.min(torch.sum(partial1 * fxu, dim=1, keepdim=False), torch.sum(partial2 * fxu, dim=1, keepdim=False))
        loss = loss + self.rewards(obs, action)
        actor_loss = -1 * loss.mean()
        actor_loss_value = actor_loss.clone().detach()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # v = self.cost_to_go(obs)
        # critic_loss = ((v - actor_loss_value) ** 2).mean()
        # self.cost_to_go_optimizer.zero_grad()
        # critic_loss.backward(inputs=list(self.cost_to_go.parameters()))
        # self.cost_to_go_optimizer.step()

        info = {'actor_loss': actor_loss.item()
                # 'terminal_cost': final_reward.mean().item(),
                # 'critic_loss': critic_loss.item()}
                }
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

        # Critic step
        critic_info = self.update_critic(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **critic_info,
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

    def supervised_train(self, batch, ):
        actor_info = self.supervised_from_mpc(batch)
        # critic_info = self.critic_step(batch, su)

        return actor_info
