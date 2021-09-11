env = dict(
    type='MountainCarContinuous-v0',
    monitor_freq=20,
    render=False,
    num_episodes_solved=100)

agent=dict(type='DQN',
    actor = dict(type='MLPNet', 
                hidden_layers=[400,300],
                act_cfg=dict(type='SiLU')),
    critic = dict(type='MLPNet', 
                hidden_layers=[400,300],
                act_cfg=dict(type='SiLU')),
    buffer= dict(type='BaseBuffer', 
                capacity=8000, 
                batch_size=256),
    actor_optimizer= dict(type='Adam',lr=1e-3),
    critic_optimizer= dict(type='Adam',lr=1e-3),
    gamma=0.995,
    explore_rate=0.1,
    network_iters=2,
    policy_noise=0.2,
    noise_clip=0.5,
    )

num_episodes=300