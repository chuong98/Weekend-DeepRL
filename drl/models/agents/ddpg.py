import numpy as np 
import torch
from torch import nn 
from mmcv.runner.optimizer import build_optimizer
from ..builder import (AGENTS, build_buffer, build_network)

class Actor(nn.Module):
    def __init__(self, network_cfg, std=0.1, noise_clip=0.3, decay_factor=0.9999):
        super().__init__()
        self.std = std
        self.noise_clip = noise_clip
        self.decay_factor=decay_factor
        self.network = build_network(network_cfg)
        self.action_act = nn.Tanh()

    def forward(self,states, add_noise=False):
        action = self.action_act(self.network(states))
        if add_noise:
            action = self.add_noise(action)
        return action 

    def add_noise(self,action):
        """
            we add Gaussian noise and clamp it in a range of values 
            supported by the environment
        """
        noise = torch.normal(torch.zeros_like(action), self.std)
        if self.noise_clip:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        return (action + noise).clamp(-1, 1)

    def decay_noise(self):
        """ reduce the noise magnitude over time"""
        self.std *=self.decay_factor

class Critic(nn.Module):
    def __init__(self, network_cfg):
        super().__init__()
        self.network = build_network(network_cfg)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.network(x)

@AGENTS.register_module()
class DDPG:
    """
        Deep Deterministic Policy Gradient.
    """
    def __init__(self, 
                num_states,
                num_actions,
                actor=dict(type='MLP'),
                critic=dict(type='MLP'),
                action_noise = dict(std=0.2,noise_clip=0.6,decay_factor=0.999),
                buffer = dict(capacity=2000, batch_size=128),
                actor_optimizer=dict(type='Adam', lr=1e-3),
                critic_optimizer=dict(type='Adam', lr=1e-3),
                gamma=0.9,
                explore_rate=0.3,
                polyak = 0.99,
                start_steps=100,
                ):
        self.num_actions = num_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # The actor and actor target network
        actor_cfg = actor.copy()
        actor_cfg['in_channels']=num_states
        actor_cfg['out_channels']=num_actions
        self.actor = Actor(actor_cfg, **action_noise).to(self.device)
        self.actor.noise_clip = None # We don't do noise clip for actor network
        self.actor_target =  Actor(actor_cfg, **action_noise).to(self.device)
        self.actor_optimizer = build_optimizer(self.actor, actor_optimizer)

        # The critic and critic target network
        critic_cfg = critic.copy()
        critic_cfg['in_channels']=num_states+num_actions
        critic_cfg['out_channels']=1
        self.critic = Critic(critic_cfg).to(self.device)
        self.critic_target = Critic(critic_cfg).to(self.device)

        # The critic and critic target twin-networks
        self.critic_optimizer = build_optimizer(self.critic, critic_optimizer)

        # The memory is used to store and replay the experience
        self.memory = build_buffer(buffer)

        # Agent parameters
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.polyak = polyak
        self.start_steps = start_steps
        self.learn_step_counter = 0
                
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self._init_weights()

    def _init_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_target_networks(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

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
        action = self.actor(input, add_noise=is_train)
        self.actor.decay_noise()
        return action.cpu().detach().numpy().flatten()

    def learn(self, state, action, reward, new_state, done):
        # Store the trainsition
        self.store_transition(state, action, reward, new_state, done)

        #sample batch from memory
        mini_batch = self.memory.getMiniBatch(device=self.device)
        (states, actions, rewards, next_states, finals) = mini_batch

        # compute the loss for the critic networks
        q_eval = self.critic(states, actions)
        with torch.no_grad():
            q_target = self.get_critic_targets(rewards, next_states, finals)
        critic_loss = self.loss_func(q_eval, q_target) 
        
        # backward and optimize the critic network 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update the target networks once every network_inters 
        # Actor Loss: We want to maximize the expected value of q_val
        q_val = self.critic(states, self.actor(states))
        actor_loss = -q_val.mean() 
        
        # backward and optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network by polyak
        self.update_target_networks()
        
        self.learn_step_counter+=1
        
    def get_critic_targets(self, rewards, next_states, finals):
        """
            Bootstrap the target 
        """
        # Predict the next actions using the target actor network
        next_actions = self.actor_target(next_states, add_noise=True)

        # The two Critic targets take each the couple (s’, a’) as input
        # and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        q_next = self.critic_target(next_states, next_actions)

        # Step 4: We get the final target of the two Critic models, 
        # which is: Qt = r + γ * q_next, where γ is the discount factor
        q_target = rewards + self.gamma* (1-finals) *q_next.squeeze()

        return q_target.unsqueeze(1) # Output [batch_size, 1]

