import random
import numpy as np 

# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.htmls
class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    The sample will be stored as a dictionary of {"state", "action", "reward", "nextState","isFinal"}.
    The memory will be used for replaying, where we return a list of samples, randomly drawed from the buffer.
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
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
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        
        self.currentPosition += 1

    def getMiniBatch(self, size) :
        indices = random.sample(np.arange(len(self.states)), 
                                k=min(size,len(self.states)))
        miniBatch = [self.getMemory(i) for i in indices]
        return miniBatch

class DQN: