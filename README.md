# Reinforcement Leaning Tutorial

## About
Weekend Deep Reinforcement Learning (DRL) is a self-study of DRL in my free time. 
DRL is very easy, especially when you already have a bit background in Control and Deep Learning. 
Even without the background, the concept is still very simple, so why not study and have fun with it.

My implementation aims to provides a minimal code with short notes for theory summary.
The code's modules are based on [MMCV](https://github.com/open-mmlab/mmcv) framework, thus very easy to adopt, adjust components by changing the config files.

Other great resource for self-study RL:
+ https://simoninithomas.github.io/deep-rl-course/#syllabus
+ https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
+ https://github.com/thu-ml/tianshou

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
Best viewed in VSCode due to latex rendering.
1. [Q-Learning](configs/QLearning/ReadMe.md): Introduction to RL with Q-Learning
2. [Deep Q-Learning](configs/DQN/ReadMe.md): 
   + Deep Q-Network (DQN - Nature 2015):  [code](drl/models/agents/dqn.py), [config](configs/DQN/dqn_mountain_car.py) 
   + Double-DQN (DDQN - AAAI 2016): [code](drl/models/agents/double_dqn.py), [config](configs/DQN/ddqn_mountain_car.py) 
   + Priority Experience Replay (ICLR 2016)
3. Actor-Critic methods:
   + [Deep Deterministic Policy Gradient (DDPG - ICLR 2016)](configs/DDPG/ReadMe.pdf): [code](drl/models/agents/ddpg.py), [config](configs/DDPG/ddpg_mountaincar_continuous.py)
   + [Twin Delayed DDPG (TD3 - ICML 2018)](configs/TD3/ReadMe.pdf): [code](drl/models/agents/td3.py), [config](configs/TD3/td3_mountaincar_continuous.py)
   + [Soft Actor-Critic (SAC - ICML 2018)](config/SAC/README.md)
4. 
### 4. Usage:

Except the first `Q-Learning` tutorial, that is for RL introduction, all other methods can be easily trained as:

```bash
python tools/train.py [path/to/config.py] [--extra_args]
```
For example, to train a Deep Q-Learning (DQN) for mountain car env, use:
```bash
python tools/train.py configs/DQN/dqn_mountain_car.py
```
The config system and modules are written based on `mmcv` [configs and registry system](https://mmcv.readthedocs.io/en/latest/understand_mmcv.html)