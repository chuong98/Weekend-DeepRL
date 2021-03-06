import argparse
import matplotlib.pyplot as plt
import gym
from gym.wrappers import Monitor
import pandas
from tqdm import tqdm
from q_learning import QLearn, build_state, to_bin

def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent')
    parser.add_argument('--episodes', '-e', type=int, default=6000)
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--monitor', '-m', action='store_true')
    parser.add_argument('--plus_reward', '-p', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = gym.make('MountainCar-v0')
    # Record the experiments
    if args.monitor:
        env = Monitor(env, './cache/mountaincar-experiment-1', force=True,
            video_callable=lambda count: count % 40 == 0)

    # Experiments parameters
    max_number_of_steps = env.spec.max_episode_steps #200
    reward_thr = env.spec.reward_threshold
    num_epochs = args.episodes

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to:
    car_position_bins = pandas.cut([-1.2, 0.6], bins=args.n_bins, retbins=True)[1][1:-1]
    car_velocity_bins = pandas.cut([-0.07, 0.07], bins=args.n_bins, retbins=True)[1][1:-1]

    def obs2state(observation):
        car_position, car_velocity = observation
        return build_state([to_bin(car_position, car_position_bins),
                            to_bin(car_velocity, car_velocity_bins)])
    # The Q-learn algorithm
    agent = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)
    reward_list = []
    # plt.ion()
    fig, ax = plt.subplots()

    for i_episode in tqdm(range(num_epochs)):
        observation = env.reset()
        episode_reward = 0
        # Discretize the observation to state
        state = obs2state(observation)

        for t in range(max_number_of_steps):
            if args.render:
                env.render()

            # Pick an action based on the current state
            action = agent.act(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            nextState = obs2state(observation)

            # modify the reward for the agent to help it learn easier
            if args.plus_reward:
                # continuous_reward = 1/2 m*v^2 + 1/2*g*h^2 
                continuous_reward = 0.5*observation[1]**2 + 0.5*9.8*observation[0]**2
                reward =reward + 0.1*continuous_reward

            agent.learn(state, action, reward, nextState)
            state = nextState
            episode_reward += reward
            reward_list.append(episode_reward)

            if done:
                break                        
        
        # Reduce the random action probability after each episode
        agent.epsilon = agent.epsilon * 0.999 # added epsilon decay
        reward_list.append(episode_reward)

    plt.plot(reward_list)
    plt.plot([0, num_epochs],[reward_thr, reward_thr])
    plt.ylabel('Acumulated Reward')
    plt.show()  