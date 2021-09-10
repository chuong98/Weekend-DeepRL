import torch.nn as nn 
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS
from mmcv.cnn import MODELS as MMCV_MODELS

BUFFERS = Registry('replayed buffer')
AGENTS = Registry('RL agent')
MODELS = Registry('models', parent=MMCV_MODELS)
NETWORKS = MODELS

def build_buffer(cfg, default_args=None):
    return build_from_cfg(cfg, BUFFERS, default_args)

def build_network(cfg):
    return NETWORKS.build(cfg)

def build_agent(cfg, default_args=None):
    return build_from_cfg(cfg, AGENTS, default_args)
    
for module in [nn.SiLU, nn.Mish]:
    ACTIVATION_LAYERS.register_module(module)