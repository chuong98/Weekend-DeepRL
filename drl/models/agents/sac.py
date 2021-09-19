import numpy as np
import torch
from torch import nn 
from torch.distributions.normal import Normal
import torch.nn.functional as F

from mmcv.runner.optimizer import build_optimizer


from ..builder import (AGENTS, build_buffer, build_network)
from .td3 import TwinCritic  

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GaussianActor(nn.Module):
    def __init__(self, network_cfg):
        super().__init__()
        self.network = build_network(network_cfg)
        self.num_actions = network_cfg['out_channels'] //2

    def forward(self,state):
        net_out = self.network(state)
        mu, log_std = net_out[:,:self.num_actions], net_out[:,-self.num_actions:]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp() 
        return mu, std 

    def action(self,state, stochastic=True):
        mu, std = self.forward(state)
        # Pre-squashed Action
        pi_dist = Normal(mu,std)
        u = pi_dist.sample() if stochastic else mu
        # Squashed action
        action = torch.tanh(u) 
        return action

    def action_with_log_prob(self,state, stochastic=True):
        mu, std = self.forward(state)
        # Pre-squashed Action
        pi_dist = Normal(mu,std)
        u = pi_dist.rsample() if stochastic else mu

        # Compute log_prob from Gaussian, and then apply correction for Tanh squashing.
        # See: appendix C in SAC paper (arXiv 1801.01290) 
        log_prob = pi_dist.log_prob(u).sum(axis=-1)
        log_prob -= (2*(np.log(2) - u - F.softplus(-2*u))).sum(axis=1)

        # Squashed action
        action = torch.tanh(u) 

        return action, log_prob


@AGENTS.register_module()
class SAC:
    def __init__(self, 
            num_states,
            num_actions,
            actor=dict(type='MLP'),
            critic=dict(type='MLP'),
            buffer = dict(capacity=2000, batch_size=128),
            actor_optimizer=dict(type='Adam', lr=1e-3),
            critic_optimizer=dict(type='Adam', lr=1e-3),
            alpha = 0.1,
            gamma=0.9,
            explore_rate=0.3,
            polyak = 0.99,
            network_iters=1,
            start_steps=100):

        self.num_actions = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # The actor network
        actor_cfg = actor.copy()
        actor_cfg['in_channels']=num_states
        actor_cfg['out_channels']=2*num_actions #(mean, std)
        self.actor = GaussianActor(actor_cfg).to(self.device)
        self.actor_optimizer = build_optimizer(self.actor, actor_optimizer)

        # The critic and critic target network
        critic_cfg = critic.copy()
        critic_cfg['in_channels']=num_states+num_actions
        critic_cfg['out_channels']=1
        self.critic = TwinCritic(critic_cfg).to(self.device)
        self.critic_target = TwinCritic(critic_cfg).to(self.device)

        # The critic and critic target twin-networks
        self.critic_optimizer = build_optimizer(self.critic, critic_optimizer)

        # The memory is used to store and replay the experience
        self.memory = build_buffer(buffer)

        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.polyak = polyak
        self.network_iters= network_iters
        self.start_steps = start_steps
        self.learn_step_counter = 0
                
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self._init_weights()

    def _init_weights(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_target_networks(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data) 

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.addMemory(state, action, reward, new_state, done)

    def act(self,state, is_train=False):
        # To improve exploration at the start of training,
        # in the first start_steps, the agent takes actions 
        # which are uniformly sampled from [-1,1]
        if is_train and (self.learn_step_counter < self.start_steps) \
            and (np.random.randn() <= self.explore_rate):# random policy
            return np.random.uniform(low=-1.0,high=1.0,size=self.num_actions)

        input = torch.Tensor(state).unsqueeze(0).to(self.device)
        action = self.actor.action(input, stochastic = is_train)
        return action.cpu().detach().numpy().flatten()

    def learn(self, state, action, reward, new_state, done):
        # Store the trainsition
        self.store_transition(state, action, reward, new_state, done)

        #sample batch from memory
        mini_batch = self.memory.getMiniBatch(device=self.device)
        (states, actions, rewards, next_states, finals) = mini_batch

        # compute the loss for the critic networks
        q1_eval, q2_eval = self.critic(states, actions)
        with torch.no_grad():
            q_target = self.get_critic_targets(rewards, next_states, finals)
        critic_loss = self.loss_func(q1_eval, q_target) + self.loss_func(q2_eval, q_target)
        
        # backward and optimize the critic network 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #update the actor and target networks once every network_inters 
        if self.network_iters==1 or self.learn_step_counter % self.network_iters ==0:
            # Actor Loss
            pred_actions, log_prob = self.actor.action_with_log_prob(states)
            q1_val,q2_val = self.critic(states, pred_actions)
            q_val = torch.min(q1_val,q2_val).squeeze() 
            # We want to maximize the q_val
            actor_loss = (self.alpha*log_prob -q_val).mean() 
            
            # backward and optimize the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network by momentum
            self.update_target_networks()

        self.learn_step_counter+=1
        
    def get_critic_targets(self, rewards, next_states, finals):
        """
            Bootstrap the target 
        """
        # Step 1: Predict the next actions using the target actor network
        next_actions, next_log_prob = self.actor.action_with_log_prob(next_states)

        # Step 2: The two Critic targets take each the couple (s’, a’) as input
        # and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        q1_target, q2_target = self.critic_target(next_states, next_actions)
        # Step 3: We pick the minimum of these two Q-values, and add the entropy
        q_target_next = torch.min(q1_target, q2_target).squeeze() 

        # Step 5: We get the final target of the two Critic models, 
        # which is: Qt = r + γ * (min(Qt1, Qt2) - alpha*log_prob(a))\
        # where γ is the discount factor
        q_target = rewards + self.gamma* (1-finals) * (q_target_next - self.alpha*next_log_prob)

        return q_target.unsqueeze(1) # Output [batch_size, 1]