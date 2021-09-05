# Reinforcement Leaning Tutorial

### 1. Env Setup:
   
```bash 
conda create -n RL --python=3.8 -y
pip install gym 
pip install gym[all] #Install the environment dependence
# or pip install cmake 'gym[atari]' scipy
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
1. [Q-Learning](Q_Learning/Q-learning.md)
2. [Deep Q-Learning (DQN)](Deep_Q_Learning/Deep_Q-learning.md)