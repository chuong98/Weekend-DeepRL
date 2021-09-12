
import numpy as np 
import torch
from torch import nn 
import torch.nn.functional as F
from mmcv.runner.optimizer import build_optimizer

from ..builder import (AGENTS, build_buffer, build_network)

class Actor(nn.Module):
    def __init__(self, network_cfg,action_mag):
        self.network = build_network(network_cfg)
        self.action_mag = action_mag

    def forward(self,states):
        logit = self.network(states)
        action = self.action_mag * logit.tanh()
        return action 

class TwinCritic(nn.Module):
    def __init__(self, network_cfg):
        super(TwinCritic, self).__init__()
        self.critic_1 = build_network(network_cfg)
        self.critic_2 = build_network(network_cfg)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.critic_1(x), self.critic_2(x)

    def forward_q1(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.critic_1(x)

@AGENTS.register_module()
class TD3:
    def __init__(self, 
                num_states,
                num_actions,
                action_range,
                actor=dict(type='MLP'),
                critic=dict(type='MLP'),
                buffer = dict(capacity=2000, batch_size=128),
                actor_optimizer=dict(type='Adam', lr=1e-3),
                critic_optimizer=dict(type='Adam', lr=1e-3),
                gamma=0.9,
                explore_rate=0.1,
                network_iters=2,
                policy_noise=0.2,
                noise_clip=0.5,
                action_mag=1,
                momentum = 0.005,
                ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # The actor and actor target network
        actor_cfg = actor.copy()
        actor_cfg['in_channels']=num_states
        actor_cfg['out_channels']=num_actions
        self.actor = Actor(actor_cfg,action_mag).to(self.device)
        self.actor_target = Actor(actor_cfg,action_mag).to(self.device)
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
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.network_iters = network_iters
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.tau = momentum
        self.action_mag = action_mag
        self.learn_step_counter = 0
                
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self._init_weights()

    def _init_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self,state):
        input = torch.Tensor(state).unsqueeze(0).to(self.device)
        action = self.actor(input)
        return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.addMemory(state, action, reward, new_state, done)

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

        #update the target networks once every network_inters 
        if self.learn_step_counter % self.network_iters ==0:
            # Actor Loss
            q_val = self.critic.forward_q1(states, self.actor(states))
            actor_loss = -q_val.mean() # We want to maximize the q_val
            
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
        next_actions = self.actor_target(next_states)

        # Step 2: We add Gaussian noise to this next action a’ 
        # and we clamp it in a range of values supported by the environment
        noise = torch.normal(torch.zeros_like(next_actions), self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions = (next_actions + noise).clamp(-self.action_mag, self.action_mag)

        # Step 3: The two Critic targets take each the couple (s’, a’) as input
        # and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        q1_target, q2_target = self.critic_target(next_states, next_actions)

        # Step 4: We pick the minimum of these two Q-values to get the target of the two Critic
        q_target_min = torch.min(q1_target, q2_target)

        # Step 5: We get the final target of the two Critic models, 
        # which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
        q_target = rewards + self.gamma* (1-finals) *q_target_min

        return q_target

    def update_target_networks(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 
