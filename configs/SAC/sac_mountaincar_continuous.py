env = dict(
    type='MountainCarContinuous-v0',
    monitor_freq=20,
    render=False,
    num_episodes_solved=100)

agent=dict(type='SAC',
    actor = dict(type='MLPNet', 
                hidden_layers=[50,30],
                act_cfg=dict(type='SiLU')),
    critic = dict(type='MLPNet', 
                hidden_layers=[50,30],
                act_cfg=dict(type='SiLU')),
    buffer= dict(type='BaseBuffer', 
                capacity=8000, 
                batch_size=256),
    actor_optimizer= dict(type='Adam',lr=1e-3),
    critic_optimizer= dict(type='Adam',lr=1e-3),
    alpha=0.05, 
    gamma=0.995,
    explore_rate=0.3,
    polyak=0.995,
    start_steps=1e4,
    )

num_episodes=300

# checkpoint saving
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None