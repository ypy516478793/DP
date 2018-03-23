"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
from new_MDP_env import MDP_env
from RL_brain import QLearningTable
from matplotlib import pyplot as plt
from collections import defaultdict


def update():
    for episode in range(3000):
        # initial observation
        observation = env.reset()
        episode_memories[episode].append(np.argmax(observation))
        r = 0

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(np.argmax(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            print("episode:", episode, "  state:", np.argmax(observation), "  action", action, "  next_state:", np.argmax(observation_), "  epsilon:", RL.epsilon)

            r += reward

            RL.store_transition(np.argmax(observation), action, reward, np.argmax(observation_))
            episode_memories[episode].append(np.argmax(observation_))

            # RL learn from this transition
            RL.learn(np.argmax(observation), action, reward, np.argmax(observation_))

            # swap observation

            observation = observation_

            # break while loop when end of this episode
            if done:
                reward_list.append(r)
                break

    # end of game
    print('game over')
    # env.destroy()

if __name__ == "__main__":
    env = MDP_env()
    n_actions = 2
    n_features = 12
    reward_list = []
    RL = QLearningTable(n_features, actions=list(range(n_actions)))
    episode_memories = defaultdict(list)
    update()
    av_reward = [np.mean(reward_list[0: i+1]) for i in range(len(reward_list))]
    plt.plot(np.arange((len(reward_list))), av_reward)
    plt.show()
    np.set_printoptions(suppress=True)
    RL.q_table['a'] = RL.q_table.idxmax(axis=1)
    print(np.hstack([np.arange(12).reshape(12, 1), RL.q_table.sort_index(axis=0, ascending=True).values]))
    print("")