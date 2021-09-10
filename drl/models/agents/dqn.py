
import numpy as np 
import torch
from torch import nn 
import torch.nn.functional as F
from mmcv.runner.optimizer import build_optimizer

from ..builder import (AGENTS, build_buffer, build_network)

@AGENTS.register_module()
class DQN:
    def __init__(self, 
                num_states,
                num_actions,
                network=dict(type='MLP'),
                buffer = dict(capacity=2000, batch_size=128),
                optimizer=dict(type='Adam', lr=1e-3),
                gamma=0.9,
                explore_rate=0.1,
                network_iters=100,
                ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network_cfg = network.copy()
        network_cfg['in_channels']=num_states
        network_cfg['out_channels']=num_actions
        # The q_network is used for calculating the current Q-value
        self.q_net = build_network(network_cfg).to(self.device)
        # The target_network is used to compute the next Q-values
        self.t_net = build_network(network_cfg).to(self.device)
        # The memory is used to store and replay the experience
        self.memory = build_buffer(buffer)

        # Agent parameters
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.network_iters = network_iters
        self.learn_step_counter = 0
        
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self.optimizer = build_optimizer(self.q_net, optimizer)

    def act(self, state):
        state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
        action_value = self.q_net(state)
        action_prob = F.softmax(action_value, dim=1).cpu().data.numpy().squeeze()
        if np.random.randn() <= self.explore_rate:# random policy
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        else: # Greedy policy
            action = np.argmax(action_prob)
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.addMemory(state, action, reward, new_state, done)

    def learn(self, state, action, reward, new_state, done):
        # Store the trainsition
        self.store_transition(state, action, reward, new_state, done)
        
        #update the target network periodically
        if self.learn_step_counter % self.network_iters ==0:
            self.t_net.load_state_dict(self.q_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        mini_batch = self.memory.getMiniBatch(device=self.device)
        (states, actions, rewards, next_states, are_final) = mini_batch

        #compute the loss
        q_eval = self.q_net(states).gather(1,actions.view(-1,1))
        q_target = self.get_target(next_states, rewards, are_final)
        loss = self.loss_func(q_eval, q_target)

        #backward and optimize the network 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_target(self, next_states, rewards, isFinals):
        with torch.no_grad():
            q_next = self.t_net(next_states)
            q_next_max = q_next.max(dim=1)[0]
            q_target = rewards + self.gamma* (1-isFinals) *q_next_max
        return q_target.unsqueeze(1) # Output [batch_size, 1]



