from torch import nn 
import torch.nn.functional as F
from ..builder import NETWORK

@NETWORK.register_module()
class MLPNetwork(nn.Module):
    """MLP Network"""
    def __init__(self, num_states, num_actions):
        super().__init__()
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