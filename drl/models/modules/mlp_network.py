from torch import nn 
from mmcv.cnn.bricks import build_activation_layer
from ..builder import NETWORKS

@NETWORKS.register_module()
class MLPNet(nn.Module):
    """MLP Network"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                hidden_layers=[50,30], 
                act_cfg=dict(type='SiLU')):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for i,channel in enumerate(hidden_layers):
            in_chans = in_channels if i==0 else hidden_layers[i-1]
            self.fc_layers.append(nn.Linear(in_chans, channel))
        self.out = nn.Linear(hidden_layers[-1],out_channels)
        self.act = build_activation_layer(act_cfg)
        # self._init_weights()

    def _init_weights(self):
        for fc in self.fc_layers():
            fc.weight.data.normal_(0,0.1)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        for fc in self.fc_layers:
            x = self.act(fc(x))
        action_logic = self.out(x)
        return action_logic