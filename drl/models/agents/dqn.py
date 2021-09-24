
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
                target_update_iters=100,
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
        self.target_update_iters = target_update_iters
        self.learn_step_counter = 0
        
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self.optimizer = build_optimizer(self.q_net, optimizer)

    def _init_weights(self):
        self.t_net.load_state_dict(self.q_net.state_dict())

    def update_target_networks(self):
        self.t_net.load_state_dict(self.q_net.state_dict())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.addMemory(state, action, reward, new_state, done)

    def act(self, state, is_train=False):
        input = torch.Tensor(state).unsqueeze(0).to(self.device)

        action_value = self.q_net(input)
        action_prob = F.softmax(action_value, dim=1).cpu().data.numpy().squeeze()
        if is_train and (np.random.randn() <= self.explore_rate):# random policy
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        else: # Greedy policy
            action = np.argmax(action_prob)
        return action

    def learn(self, state, action, reward, new_state, done):
        # Store the trainsition
        self.store_transition(state, action, reward, new_state, done)
        
        #sample batch from memory
        mini_batch = self.memory.getMiniBatch(device=self.device)
        (states, actions, rewards, next_states, finals) = mini_batch

        #compute the loss
        q_eval = self.q_net(states).gather(1,actions.long().view(-1,1))
        with torch.no_grad():
            q_target = self.get_targets(rewards, next_states, finals)
        loss = self.loss_func(q_eval, q_target)

        #backward and optimize the network 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update the target network periodically
        if self.learn_step_counter % self.target_update_iters ==0:
            self.update_target_networks()
        self.learn_step_counter+=1

    def get_targets(self, rewards, next_states, finals):
        """
            Bootstrap the target 
        """ 
        q_next = self.t_net(next_states)
        q_next_max = q_next.max(dim=1)[0]
        q_target = rewards + self.gamma* (1-finals) *q_next_max
        return q_target.unsqueeze(1) # Output [batch_size, 1]



