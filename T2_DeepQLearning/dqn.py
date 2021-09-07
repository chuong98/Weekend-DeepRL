import random
import numpy as np 
import torch
from torch import nn 
import torch.nn.functional as F

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    The sample will be stored as a dictionary of {"state", "action", "reward", "nextState","isFinal"}.
    The memory will be used for replaying, where we return a list of samples, randomly drawed from the buffer.
    """
    def __init__(self, size):
        self.size = size
        self.counter = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return dict(state = self.states[index],
                    action= self.actions[index],
                    reward= self.rewards[index], 
                    newState=self.newStates[index],
                    isFinal=self.finals[index])

    def addMemory(self, state, action, reward, newState, isFinal) :
        if (self.counter >= self.size - 1) :
            self.counter = 0
        if (len(self.states) > self.size) :
            self.states[self.counter] = state
            self.actions[self.counter] = action
            self.rewards[self.counter] = reward
            self.newStates[self.counter] = newState
            self.finals[self.counter] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        
        self.counter += 1

    def getMiniBatch(self, size) :
        indices = random.sample(np.arange(len(self.states)), 
                                k=min(size,len(self.states)))
        mini_batch = [self.getMemory(i) for i in indices]
        return dict(
            states = torch.FloatTensor([m['state'] for m in mini_batch]),
            actions = torch.LongTensor([m['action'] for m in mini_batch]),
            rewards = torch.FloatTensor([m['reward'] for m in mini_batch]),
            next_states = torch.FloatTensor([m['newState'] for m in mini_batch]),
            is_finals = torch.FloatTensor([m['isFinal'] for m in mini_batch])
        )

class Network(nn.Module):
    """MLP Network"""
    def __init__(self, num_states, num_actions):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,num_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = F.silu(x)
        action_logic = self.out(x)
        return action_logic

class DQN:
    def __init__(self, 
                    num_states, 
                    num_actions,
                    train_cfg=dict( batch_size=128, 
                                    optim_lr=1e-2,
                                    gamma=0.9,
                                    explore_rate=0.1,
                                    memory_size=2000,
                                    network_iters=100)):
        self.num_actions = num_actions
        # The eval network is used for calculating the current Q-value
        self.net = Network(num_states, num_actions)
        # The target network is used to compute the next Q-values
        self.target_net = Network(num_states,num_actions)
        self.batch_size = train_cfg['batch_size']
        self.gamma = train_cfg['gamma']
        self.explore_rate = train_cfg['explore_rate']
        self.network_iters = train_cfg['network_iters']
        self.memory = Memory(train_cfg['memory_size'])
        self.learn_step_counter = 0
        # Network optimizer
        self.loss_func = nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=train_cfg['optim_lr'])

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if torch.cuda.is_available():
            state = state.cuda()
        action_value = self.net(state)
        action_prob = F.softmax(action_value, dim=1).cpu().data.numpy()
        if np.random.randn() <= self.explore_rate:# random policy
            action = np.random.choice(np.arange(self.num_actions),
                                      p=action_prob)
        else: # Greedy policy
            action = np.argmax(action_prob)
        return action

    def store_transition(self, state, action, reward, new_state, is_final):
        self.memory.addMemory(state, action, reward, new_state, is_final)

    def learn(self):
        #update the target network periodically
        if self.learn_step_counter % self.network_iters ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        mini_batch = self.memory.getMiniBatch(self.batch_size)
        states = mini_batch['states']
        rewards = mini_batch['rewards']
        next_states = mini_batch['next_states']
        isFinals = mini_batch['is_finals']

        #compute the loss
        if torch.cuda.is_available():
            states.cuda()
            rewards.cuda()
            next_states.cuda()
            isFinals.cuda()
        q_eval = self.net(states)
        with torch.no_grad():
            q_next = self.target_net(next_states)
            q_next_max = torch.max(q_next, dim=1)[0].view(-1,1)
            q_target = rewards + self.gamma* (1-isFinals) *q_next_max
        loss = self.loss_func(q_eval, q_target)

        #backward and optimize the network 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

