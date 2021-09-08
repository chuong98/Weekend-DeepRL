# Reinforcement Leaning Tutorial

This is my personal notes for the course [Udemy Deep Reinforcement Learning 2.0](https://www.udemy.com/course/deep-reinforcement-learning/).

In addition, I also update the code from [Basic Reinforcement Learning](https://github.com/vmayoral/basic_reinforcement_learning), using Pytorch instead of Keras.

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
1. [Q-Learning](T1_QLearning/Q-learning.md)
2. [Deep Q-Learning (DQN)](T2_DeepQLearning/DQN.md)