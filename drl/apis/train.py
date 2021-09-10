import random
import warnings
import statistics
import numpy as np
import torch
import os.path as osp 

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
    if 'seed' in cfg.keys(): env.seed(cfg.seed) 
    
    # Record the experiments
    if cfg.env.get('monitor_freq',0) >0:
        env = Monitor(env, osp.join(cfg.work_dir,'monitor'), 
                    force=True,
                    video_callable=lambda count: count % cfg.env.monitor_freq == 0)
    
    # Build agent
    is_discrete = isinstance(env.action_space.sample(), int) # Is Action Space discrete 
    cfg.agent.num_actions = env.action_space.n if is_discrete else env.action_space.shape[0]
    cfg.agent.num_states = env.observation_space.shape[0]
    agent = build_agent(cfg.agent)

    # Experiments parameters
    max_number_of_steps = env.spec.max_episode_steps #200
    solved_reward_thr = env.spec.reward_threshold
    num_episodes = cfg.num_episodes
    reward_list = []

    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_number_of_steps):
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
        
        reward_list.append(episode_reward)
        # Consider the problem is solved if getting average reward 
        # above the threshold over 100 consecutive trials.
        num_episodes_solved = cfg.env.get('num_episodes_solved',100)
        if num_episodes_solved > 0 and i_episode > num_episodes_solved:
            mean_reward = sum(reward_list[-num_episodes_solved:])/num_episodes_solved
            if mean_reward > solved_reward_thr: 
                break

        # TODO: Add save checkpoints, logs

    mean_reward = sum(reward_list[-num_episodes_solved:])/num_episodes_solved
    print(f"Finished at Episode:{i_episode} - with mean reward: {mean_reward}")
    env.close()