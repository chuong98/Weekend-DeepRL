import gym
from gym.wrappers import Monitor
import numpy
import pandas
from tqdm import tqdm
from q_learning import QLearn, build_state, to_bin


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    # Record the experiments
    env = Monitor(env, './cache/mountaincar-experiment-1', force=True,
        video_callable=lambda count: count % 40 == 0)

    # Experiments parameters
    num_episode = 6000
    max_number_of_steps = env.spec.max_episode_steps #200
    n_bins = 10

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to:
    car_position_bins = pandas.cut([-1.2, 0.6], bins=n_bins, retbins=True)[1][1:-1]
    car_velocity_bins = pandas.cut([-0.07, 0.07], bins=n_bins, retbins=True)[1][1:-1]

    def obs2state(observation):
        car_position, car_velocity = observation
        return build_state([to_bin(car_position, car_position_bins),
                            to_bin(car_velocity, car_velocity_bins)])
    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)
    cumulated_reward = 0

    for i_episode in tqdm(range(num_episode)):
        observation = env.reset()
        # Discretize the observation to state
        state = obs2state(observation)

        for t in range(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            nextState = obs2state(observation)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward
                
            if reward !=-1:
                print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))
            if done:
                break                        
        
        # Reduce the random action probability after each episode
        qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay