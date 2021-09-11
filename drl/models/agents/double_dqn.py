
import torch
from ..builder import AGENTS
from .dqn import DQN 

@AGENTS.register_module()
class DDQN(DQN):
    """ Double DQN """
    def get_target(self, rewards, next_states, finals):
        # action selection: using the eval network
        q_next  = self.q_net(next_states)
        action_max_idxes = q_next.argmax(dim=1).view(-1,1)
        # action evaluation: using target network
        q_next = self.t_net(next_states) 
        q_next_max = q_next.gather(1,action_max_idxes)
        # target
        q_target = rewards + self.gamma* (1-finals) *q_next_max.squeeze()
        return q_target.unsqueeze(1) # Output [batch_size, 1]