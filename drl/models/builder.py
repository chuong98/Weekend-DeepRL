from mmcv.utils import Registry, build_from_cfg

BUFFER = Registry('replayed buffer')
NETWORK = Registry('deep neural network')
AGENT = Registry('RL agent')

def build_buffer(cfg, default_args=None):
    return build_from_cfg(cfg, BUFFER, default_args)

def build_network(cfg, default_args=None):
    return build_from_cfg(cfg, NETWORK, default_args)

def build_agent(cfg, default_args=None):
    return build_from_cfg(cfg, AGENT, default_args)

