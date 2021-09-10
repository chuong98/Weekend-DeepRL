env = dict(
    type='MountainCar-v0',
    monitor_freq=20,
    render=False,
    num_episodes_solved=100)

agent=dict(type='DQN',
    network=dict(type='MLPNet', 
                hidden_layers=[50,30],
                act_cfg=dict(type='SiLU')),
    buffer= dict(type='BaseBuffer', 
                capacity=8000, 
                batch_size=256),
    optimizer= dict(type='Adam',lr=1e-3),
    gamma=0.995,
    explore_rate=0.1,
    network_iters=100            
    )

num_episodes=2000

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]