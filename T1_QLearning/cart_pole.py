import argparse
import matplotlib.pyplot as plt
import gym
from gym.wrappers import Monitor
import pandas
from tqdm import tqdm
from q_learning import QLearn, build_state, to_bin

def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent')
    parser.add_argument('--episodes', '-e', type=int, default=500)
    parser.add_argument('--n_bins_pos', type=int, default=10)
    parser.add_argument('--n_bins_angle', type=int, default=8)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--monitor', '-m', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = gym.make('CartPole-v0')
    # Record the experiments
    if args.monitor:
        env = Monitor(env, './cache/cartpole-experiment-1', force=True,
                    video_callable=lambda count: count % 20 == 0)

    # Experiments parameters
    max_number_of_steps = env.spec.max_episode_steps #200
    reward_thr = env.spec.reward_threshold
    num_episodes = args.episodes

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to:
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=args.n_bins_pos, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=args.n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=args.n_bins_pos, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=args.n_bins_angle, retbins=True)[1][1:-1]

    def obs2state(observation):
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        return build_state([to_bin(cart_position, cart_position_bins),
                            to_bin(pole_angle, pole_angle_bins),
                            to_bin(cart_velocity, cart_velocity_bins),
                            to_bin(angle_rate_of_change, angle_rate_bins)])
    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=1.0, epsilon=0.15)
    
    reward_list = []
    for i_episode in tqdm(range(num_episodes)):
        observation = env.reset()
        acumulated_reward = 0
        # Discretize the observation to state
        state = obs2state(observation)

        for t in range(max_number_of_steps):
            if args.render:
                env.render()

            # Pick an action based on the current state
            action = qlearn.choose_action(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            nextState = obs2state(observation)
            acumulated_reward += reward
            qlearn.learn(state, action, reward, nextState)
            state = nextState
            
            if done:
                break                        
        
        reward_list.append(acumulated_reward)
    
    plt.plot(reward_list)
    plt.plot([0, num_episodes],[reward_thr, reward_thr])
    plt.ylabel('Acumulated Reward')
    plt.show()    
