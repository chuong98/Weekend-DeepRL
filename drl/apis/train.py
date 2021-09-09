import random
import warnings
import statistics
import numpy as np
import torch

import gym
from gym.wrappers import Monitor
from tqdm import tqdm
from drl.models import build_agent

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_agent(cfg):
    # Build Environment
    env = gym.make(cfg.env.type)
    # Record the experiments
    if cfg.env.get('monitor_freq',0) >0:
        env = Monitor(env, f'./cache/{cfg.env.type}_{cfg.agent.type}', 
                    force=True,
                    video_callable=lambda count: count % cfg.env.monitor_freq == 0)
    
    # Build agent
    cfg.agent.num_actions = env.action_space.n
    cfg.agent.num_states = env.observation_space.shape[0]
    agent = build_agent(cfg.agent)
    
    # Experiments parameters
    max_number_of_steps = env.spec.max_episode_steps #200
    solved_reward_thr = env.spec.reward_threshold
    num_episodes = cfg.train_cfg.num_episodes
    reward_list = []

    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0

        for t in range(max_number_of_steps):
            if cfg.env.render: env.render()

            # Pick an action based on the current state
            action = agent.act(state)
            
            # Execute the action and get feedback
            new_state, reward, done, info = env.step(action)

            # Learn with new experience
            agent.learn(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward

            if done:
                break                        
        
        # Consider the problem is solved if getting average reward 
        # above the threshold over 100 consecutive trials.
        num_episodes_solved = cfg.env.get('num_episodes_solved',100)
        mean_reward = statistics.mean(reward_list[-num_episodes_solved:])
        if mean_reward > solved_reward_thr: 
            print(f"Solved at Episode:{i_episode} - with mean reward: {mean_reward}")
            break

        # TODO: Add save cfg, checkpoints