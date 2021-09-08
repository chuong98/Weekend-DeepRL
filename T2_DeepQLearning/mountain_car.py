import argparse
import matplotlib.pyplot as plt
import statistics
import gym
from gym.wrappers import Monitor
from tqdm import tqdm
from dqn import DQN, DDQN

def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--monitor', '-m', action='store_true')
    parser.add_argument('--model', '-M', type=str, default='ddqn')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = gym.make('MountainCar-v0')
    # Record the experiments
    if args.monitor:
        env = Monitor(env, './cache/mountaincar-v0-dqn', force=True,
            video_callable=lambda count: count % 20 == 0)

    # Experiments parameters
    max_number_of_steps = env.spec.max_episode_steps #200
    reward_thr = env.spec.reward_threshold
    num_episodes = args.episodes

    # The DQN algorithm
    model_args = dict(num_states=env.observation_space.shape[0],
                    num_actions=env.action_space.n,
                    batch_size=256,
                    optim_lr=1e-3,
                    gamma=0.995,
                    explore_rate=0.1,
                    memory_size=8000,
                    network_iters=100)
    if args.model == 'ddqn':
        agent = DDQN(**model_args)
    else:
        agent = DQN(**model_args)
    reward_list = []
    fig, ax = plt.subplots()

    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0

        for t in range(max_number_of_steps):
            if args.render: env.render()

            # Pick an action based on the current state
            action = agent.choose_action(state)
            # Execute the action and get feedback
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward
            reward_list.append(episode_reward)

            if done:
                break                        
        
        # Reduce the random action probability after each episode
        reward_list.append(episode_reward)

        # Consider the problem is solved if getting average reward 
        # of -110.0 over 100 consecutive trials.
        mean_reward = statistics.mean(reward_list[-100:])
        if mean_reward > -70: #-110.0: is still bad visually
            print(f"Solved - with mean reward: {mean_reward}")
            break
    env.close()
    plt.plot(reward_list)
    plt.plot([0, num_episodes],[reward_thr, reward_thr])
    plt.ylabel('Acumulated Reward')
    plt.show()  