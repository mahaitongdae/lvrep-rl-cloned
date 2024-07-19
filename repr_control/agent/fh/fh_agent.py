from repr_control.agent.sac.sac_agent import SACAgent
from repr_control.agent.rfsac.rfsac_agent import RFVCritic, nystromVCritic
import torch
import copy
from repr_control.utils.util import unpack_batch
import torch.nn.functional as F

class CustomModelRFSACAgent(SACAgent):

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
            action_range=action_range,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            device=device,
            **kwargs
        )

        if use_nystrom == False:  # use RF
            self.critic = RFVCritic(s_dim=state_dim, sigma=sigma, rf_num=rf_num, learn_rf=learn_rf, **kwargs).to(self.device)
        else:  # use nystrom
            feat_num = rf_num
            self.critic = nystromVCritic(sigma=sigma, feat_num=feat_num, buffer=replay_buffer, learn_rf=learn_rf,
                                         **kwargs).to(self.device)
        # self.critic = Critic().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999])
        self.args = kwargs
        self.dynamics = dynamics_fn
        self.reward_fn = rewards_fn

    def get_reward(self, state, action):
        reward = self.reward_fn(state, action)
        return torch.reshape(reward, (reward.shape[0], 1))

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        # dist = self.actor(batch.state, batch.next_state)
        dist = self.actor(batch.state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        reward = self.get_reward(batch.state, action)  # use reward in q-fn
        q1, q2 = self.critic(self.dynamics(batch.state, action))
        q = self.discount * torch.min(q1, q2) + reward

        actor_loss = ((self.alpha) * log_prob - q).mean()

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

    def critic_step(self, batch):
        """
        Critic update step
        """
        # state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
        state, action, next_state, reward, done = unpack_batch(batch)

        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
            next_q1, next_q2 = self.critic_target(self.dynamics(next_state, next_action))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            next_reward = self.get_reward(next_state, next_action)  # reward for new s,a
            target_q = next_reward + (1. - done) * self.discount * next_q

        q1, q2 = self.critic(self.dynamics(state, action))
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        info = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'layer_norm_weights_norm': self.critic.norm.weight.norm(),
        }

        dist = {
            'td_error': (torch.min(q1, q2) - target_q).cpu().detach().clone().numpy(),
            'q': torch.min(q1, q2).cpu().detach().clone().numpy()
        }

        info.update({'critic_dist': dist})

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        batch = buffer.sample(batch_size)

        # Acritic step
        critic_info = self.critic_step(batch)
        # critic_info = self.rfQcritic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **critic_info,
            **actor_info,
        }
