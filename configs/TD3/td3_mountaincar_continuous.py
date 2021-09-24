_base_='../DDPG/ddpg_mountaincar_continuous.py'

agent=dict(type='TD3',
        target_update_iters=2,
        )