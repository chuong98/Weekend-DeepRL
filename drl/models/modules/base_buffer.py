import numpy as np 
import torch
from operator import itemgetter
from ..builder import BUFFERS

@BUFFERS.register_module()
class BaseBuffer(object):
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    The sample will be stored as a dictionary of {"state", "action", "reward", "nextState","isFinal"}.
    The memory will be used for replaying, where we return a list of samples, randomly drawed from the buffer.
    """
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.counter = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []
        self.batch_size = batch_size

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return dict(state = self.states[index],
                    action= self.actions[index],
                    reward= self.rewards[index], 
                    newState=self.newStates[index],
                    isFinal=self.finals[index])

    def addMemory(self, state, action, reward, newState, isFinal, **kwargs) :
        if (self.counter >= self.capacity - 1) :
            self.counter = 0
        if (len(self.states) > self.capacity) :
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

    def getMiniBatch(self, device='cpu') :
        current_size = self.getCurrentSize()
        if self.batch_size < current_size :
            indices = np.random.randint(0,current_size, self.batch_size)
        else:
            indices = np.arange(current_size)
        return (torch.FloatTensor(itemgetter(*indices)(self.states), device=device),
                torch.LongTensor(itemgetter(*indices)(self.actions), device=device),
                torch.FloatTensor(itemgetter(*indices)(self.rewards), device=device),
                torch.FloatTensor(itemgetter(*indices)(self.newStates), device=device),
                torch.FloatTensor(itemgetter(*indices)(self.finals), device=device))