import gym
from gym.wrappers import Monitor
import numpy
import pandas
from tqdm import tqdm
from q_learning import QLearn, build_state, to_bin

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # Record the experiments
    env = Monitor(env, '/tmp/cartpole-experiment-1', force=True,
        video_callable=lambda count: count % 20 == 0)

    # Experiments parameters
    num_episode = 1200
    max_number_of_steps = env.spec.max_episode_steps #200
    last_time_steps = numpy.ndarray(0)
    n_bins_pos = 8
    n_bins_angle = 10

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to:
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins_pos, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins_pos, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    def obs2state(observation):
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        return build_state([to_bin(cart_position, cart_position_bins),
                            to_bin(pole_angle, pole_angle_bins),
                            to_bin(cart_velocity, cart_velocity_bins),
                            to_bin(angle_rate_of_change, angle_rate_bins)])
    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)
    
    for i_episode in tqdm(range(num_episode)):
        observation = env.reset()
        # Discretize the observation to state
        state = obs2state(observation)

        for t in range(max_number_of_steps):
            env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            nextState = obs2state(observation)

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break                        

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print()
    # print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()