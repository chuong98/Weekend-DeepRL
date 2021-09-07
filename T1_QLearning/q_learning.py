import random
import numpy 

# Some helper functions to quantize observation into discrete states.
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def getQ(self,state,action):
        return self.q.get((state,action),0.0)
    
    def choose_action(self, state, return_q=False):
        # Get quality of all action at the current state
        q = [self.getQ(state, a) for a in self.actions]
        # Get the highest q
        maxQ = max(q)

        # Take random action by chance of epsilon
        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + (random.random() - .5) * mag 
                    for i in range(len(self.actions))] 
            maxQ = max(q)

        # In case there're several state-action max values 
        # we select a random one among them
        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q:
            return action, q
        return action

    def learn(self, state, action, reward, new_state):
        '''
        Q-learning:
            td_target = reward + self.gamma*max(Q(new_state))
            Q(s, a) += alpha * (td_target - Q(s,a))            
        '''
        max_qnew = max([self.getQ(new_state, a) for a in self.actions])
        td_target = reward + self.gamma*max_qnew
        self.learnQ(state, action, reward, td_target)
    
    def learnQ(self, state, action, reward, td_target):
        '''
            Update Q function:
                Q(s, a) += alpha * (td_target - Q(s,a))             
        '''
        old_q = self.q.get((state, action), None)
        if old_q is None:
            self.q[(state, action)] = reward    
        else:
            self.q[(state, action)] = old_q + self.alpha * (td_target - old_q)