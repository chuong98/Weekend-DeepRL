env = dict(
    type='MountainCar-v0',
    monitor_freq=20,
    render=False,
    num_episodes_solved=100)

agent=dict(type='DQN',
    network=dict(type='MLPNetwork', 
                channels=[100,50],
                activation_cfg='silu'),
    buffer= dict(type='BaseBuffer', 
                capacity=8000, 
                batch_size=256),
    optim_lr=1e-3,
    gamma=0.995,
    explore_rate=0.1,
    network_iters=100            
    )