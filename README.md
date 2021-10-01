# Reinforcement Leaning Tutorial

## About
Weekend Deep Reinforcement Learning (DRL) is a self-study of DRL in my free time. 
DRL is very easy, especially when you already have a bit background in Control and Deep Learning. 
Even without the background, the concept is still very simple, so why not study and have fun with it.

My implementation aims to provides a minimal code implementation, and short notes to summarize the theory.
+ The code, modules, and config system are written based on `mmcv` [configs and registry system](https://mmcv.readthedocs.io/en/latest/understand_mmcv.html), thus very easy to adopt, adjust components by changing the config files.
+ Lecture Notes: No lengthy math, just the motivation concept, key equations for implementing, and a summary of tricks that makes the methods work. More important, I try to make the connection with previous methods as possible. 

Following are the great resource that I learn from:
+ https://spinningup.openai.com/en/latest/
+ https://simoninithomas.github.io/deep-rl-course/#syllabus
+ https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
+ https://github.com/DLR-RM/stable-baselines3
+ https://github.com/thu-ml/tianshou
+ https://github.com/araffin/rl-baselines-zoo
+ https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
+ https://intellabs.github.io/coach/usage.html#
### 1. Env Setup:
   
```bash 
conda create -n RL --python=3.8 -y
conda install tqdm mathplotlib scipy
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install gym 
pip install gym[all] #Install the environment dependence
# or pip install cmake 'gym[atari]'
pip install pybullet
``` 

### 2. Try Gym environment
   
```python
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset() # Before start, reset the environment 
    for t in range(100):
        env.render()            
        print(observation)
        action = env.action_space.sample() # This is where your code should return action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

+ Every environment comes with an `env.action_space` and an `env.observation_space`.
+ List all available environments: `gym.envs.registry.all()`.

### 3. Algorithms:

1. [Q-Learning](configs/QLearning/ReadMe.md): Introduction to RL with Q-Learning
2. [Deep Q-Learning](configs/DQN/ReadMe.md): 
   + [Deep Q-Network (DQN - Nature 2015)](https://www.nature.com/articles/nature14236):  [code](drl/models/agents/dqn.py), [config](configs/DQN/dqn_mountain_car.py) 
   + [Double-DQN (DDQN - AAAI 2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847): [code](drl/models/agents/double_dqn.py), [config](configs/DQN/ddqn_mountain_car.py) 
   + [Dueling DQN (DuelDQN - ICLM 2016)](http://proceedings.mlr.press/v48/wangf16.pdf)
3. Actor-Critic methods:
   + [Deep Deterministic Policy Gradient (DDPG - ICLR 2016)](https://arxiv.org/abs/1509.02971): [Note](configs/DDPG/ReadMe.pdf), [code](drl/models/agents/ddpg.py), [config](configs/DDPG/ddpg_mountaincar_continuous.py)
   + [Twin Delayed DDPG (TD3 - ICML 2018)](https://arxiv.org/abs/1802.09477): [Note](configs/TD3/ReadMe.pdf), [code](drl/models/agents/td3.py), [config](configs/TD3/td3_mountaincar_continuous.py)
   + [Soft Actor-Critic (SAC - ICML 2018)](https://arxiv.org/abs/1812.05905): [Note](config/SAC/README.md), [code](drl/models/agents/sac.py), [config](configs/TD3/sac_mountaincar_continuous.py)
   + [Meta-SAC](https://arxiv.org/abs/2007.01932)
   + [Smooth Exploration for Robotic Reinforcement Learning]
4. Policy Gradient:
   + Recap and overview of the methods
   + Vanilla Policy Gradient 
   + Trust Region Policy Optimization (TRPO)
   + Proximal Policy Optimization (PP0).
5. How to deal with Sparse Reward for Off-Line learning:
   + [Priority Experience Replay (ICLR 2016)](https://arxiv.org/abs/1511.05952)
   + [Hindsight Experience Replay (NIPS 2017)](https://arxiv.org/abs/1707.01495) 
   + [First Return, Then Explore (Nature 2020)](https://arxiv.org/abs/2004.12919)
6. On-Line Policy  

### 4. Usage:

Except the first `Q-Learning` tutorial, that is for RL introduction, all other methods can be easily trained as:

```bash
python tools/train.py [path/to/config.py] [--extra_args]
```
For example, to train a Deep Q-Learning (DQN) for mountain car env, use:
```bash
python tools/train.py configs/DQN/dqn_mountain_car.py
```
