import numpy as np
import tensorflow as tf
from collections import defaultdict

n_state = 6

lr = np.zeros(n_state)
hr = np.zeros(n_state)
for i in range(3000):
    if reward_list[i] < 0.5:
        for element in set(episode_memories[i]):
            lr[element] += 1
    else:
        for element in set(episode_memories[i]):
            hr[element] += 1

lr2 = np.zeros([n_state, n_state])
hr2 = np.zeros([n_state, n_state])
for i in range(3000):
    key = set()
    if reward_list[i] < 0.5:
        for i1 in range(len(episode_memories[i])-1):
            for i2 in range(i1+1, len(episode_memories[i])):
                new_key = tuple([episode_memories[i][i1], episode_memories[i][i2]])
                if new_key not in key: lr2[new_key] += 1
                key.add(new_key)
    else:
        for i1 in range(len(episode_memories[i])-1):
            for i2 in range(i1+1, len(episode_memories[i])):
                new_key = tuple([episode_memories[i][i1], episode_memories[i][i2]])
                if new_key not in key: hr2[new_key] += 1
                key.add(new_key)

d = defaultdict(float)
for i in critic.q_table_o1.index:
    l = int(i.strip('[]').split(',')[0])
    for k in critic.q_table_o1.columns:
        d[(l,k)] = critic.q_table_o1.loc[i, k]  # key word: (state, option o)
import operator
sort_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)