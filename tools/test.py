from drl.models.builder import NETWORKS, build_network

network=dict(type='MLPNet', 
        in_channels=2,out_channels=3,
        hidden_layers=[50,30],
        act_cfg=dict(type='SiLU'))
import pdb; pdb.set_trace()
net = build_network(network)