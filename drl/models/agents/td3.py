import torch
from torch import nn 
from mmcv.runner.optimizer import build_optimizer

from ..builder import (AGENTS, build_network)
from .ddpg import DDPG

class TwinCritic(nn.Module):
    def __init__(self, network_cfg):
        super(TwinCritic, self).__init__()
        self.critic_1 = build_network(network_cfg)
        self.critic_2 = build_network(network_cfg)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.critic_1(x), self.critic_2(x)

    def q1(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.critic_1(x)



@AGENTS.register_module()
class TD3(DDPG):
    def __init__(self, 
                    num_states,
                    num_actions,
                    network_iters=2,
                    *args,
                    **kwargs): 
        super().__init__(num_states, num_actions, *args,**kwargs)
        self.network_iters= network_iters

        # TD3 only changes the Critic Network 
        critic_cfg = kwargs['critic'].copy()
        critic_cfg['in_channels']=num_states+num_actions
        critic_cfg['out_channels']=1
        self.critic = TwinCritic(critic_cfg).to(self.device)
        self.critic_target = TwinCritic(critic_cfg).to(self.device)

        # The critic and critic target twin-networks
        self.critic_optimizer = build_optimizer(self.critic, kwargs['critic_optimizer'])
        # Since we overwrote the critic network, we need to reinitilize the weights
        self._init_weights()

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
        if self.network_iters==1 or self.learn_step_counter % self.network_iters ==0:
            # Actor Loss
            q_val = self.critic.q1(states, self.actor(states))
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
        next_actions = self.actor_target(next_states, add_noise=True)

        # Step 2: The two Critic targets take each the couple (s’, a’) as input
        # and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        q1_next, q2_next = self.critic_target(next_states, next_actions)

        # Step 3: We pick the minimum of these two Q-values to get the target of the two Critic
        q_next = torch.min(q1_next, q2_next).squeeze()
        
        # Step 4: We get the final target of the two Critic models, 
        # which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
        q_target = rewards + self.gamma* (1-finals) *q_next

        return q_target.unsqueeze(1) # Output [batch_size, 1]

