import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
# from MDP_env import MDP_env
from new_MDP_env import MDP_env
from collections import defaultdict

# seed for reproducible
np.random.seed(2)
tf.set_random_seed(2)

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic
N_F = 12  # number of features
N_A = 2  # number of actions
N_O = (1, 2)  # number of options

# Class definition
class Actor(object):
    def __init__(self, sess, n_features, n_options, n_actions, lr=0.001):
        self.sess = sess
        self.epsilon = np.ones(2)
        self.n_options = n_options
        self.chosen_options = np.zeros(len(n_options), dtype=np.int32)
        self.new_features = n_features + n_options[1]
        self.ep_info = defaultdict(list)

        self.s = tf.placeholder(tf.float32, [None, self.new_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=40,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )



        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train_actor'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def store_transition(self, sv, a):
        self.ep_info['ep_sv'].append(sv)
        self.ep_info['ep_a'].append(a)

    def learn_ep(self, error_list):
        for i in range(len(error_list)):
            self.learn(self.ep_info['ep_sv'][i], self.ep_info['ep_a'][i], error_list[i])
        self.ep_info['ep_sv'], self.ep_info['ep_a'] = [], []

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int

    def eval_options_probs(self, Q, l):
        options_probs = np.ones(self.n_options[l]) * self.epsilon[l] / self.n_options[l]
        options_probs[np.argmax(Q)] += 1 - self.epsilon[l]
        return options_probs

    def choose_option(self, l, Q):
        options_probs = self.eval_options_probs(Q, l)
        self.chosen_options[l] = np.random.choice(np.arange(options_probs.size), p=options_probs)
        # return self.chosen_options[l]

    def eval_option_term(self, l, Q_):
        options_probs_next = self.eval_options_probs(Q_, l)
        options_term_prob_next = 1 - options_probs_next
        return options_term_prob_next[self.chosen_options[l]]

    def anneal(self, layer):
        if self.epsilon[layer] > 0.1:
            if layer == 0:
                self.epsilon[layer] = self.epsilon[layer] - 0.001
            elif layer == 1:
                self.epsilon[layer] = self.epsilon[layer] - 0.001




class Critic(object):
    def __init__(self, n_options, lr=0.01, reward_decay=0.99):
        self.options0 = list(range(n_options[0]))  # a list
        self.options1 = list(range(n_options[1]))
        self.lr = lr
        self.gamma = reward_decay
        self.q_table_o0 = pd.DataFrame(columns=self.options0, dtype=np.float64)
        self.q_table_o1 = pd.DataFrame(columns=self.options1, dtype=np.float64)
        self.ep_info_0 = defaultdict(list)
        self.ep_info_1 = defaultdict(list)

    def learn_ep(self, l, v_, o):
        error = []
        if l == 1:
            discounted_ep_rs = self.discount_rewards(self.ep_info_1['ep_r'])
            for i in range(len(self.ep_info_1['ep_r'])):
                error.append(self.learn(1, self.ep_info_1['ep_s'][i], discounted_ep_rs[i], v_, o))
            self.ep_info_1['ep_s'], self.ep_info_1['ep_r'] = [], []
        else:
            discounted_ep_rs = self.discount_rewards(self.ep_info_0['ep_r'])
            for i in range(len(self.ep_info_0['ep_r'])):
                error.append(self.learn(0, self.ep_info_0['ep_s'][i], discounted_ep_rs[i], v_, o))
            self.ep_info_0['ep_s'], self.ep_info_0['ep_r'] = [], []
        return error

    def learn(self, l, s, r, v_, o):
        q_target = r + v_
        if l == 1:
            q_predict = self.q_table_o1.loc[s, o]
            self.q_table_o1.loc[s, o] += self.lr * (q_target - q_predict)  # update
        else:
            q_predict = self.q_table_o0.loc[s, o]
            self.q_table_o0.loc[s, o] += self.lr * (q_target - q_predict)  # update
        error = q_target - q_predict
        return error


    def check_state_exist(self, state, layer):
        if layer == 0:
            if state not in self.q_table_o0.index:
                # append new state to q table
                self.q_table_o0 = self.q_table_o0.append(
                    pd.Series(
                        [0] * len(self.options0),
                        index=self.q_table_o0.columns,
                        name=state,
                    )
                )
        if layer == 1:
            if state not in self.q_table_o1.index:
                # append new state to q table
                self.q_table_o1 = self.q_table_o1.append(
                    pd.Series(
                        [0] * len(self.options1),
                        index=self.q_table_o1.columns,
                        name=state,
                    )
                )

    def eval_q(self, state, layer, chosen_option=None):
        self.check_state_exist(state, layer)
        if layer == 0:
            Q_options = self.q_table_o0.loc[state].values
        else:
            Q_options = self.q_table_o1.loc[state].values
        # Q_options = np.ravel(Q_options)
        if chosen_option is None:
            return Q_options
        else:
            return Q_options[chosen_option]

    def store_transition(self, layer, s, r):
        if layer == 0:
            self.ep_info_0['ep_s'].append(s)
            self.ep_info_0['ep_r'].append(r)
        else:
            self.ep_info_1['ep_s'].append(s)
            self.ep_info_1['ep_r'].append(r)

    def discount_rewards(self, ep_rs):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(ep_rs)
        running_add = 0
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * GAMMA + ep_rs[t]
            discounted_ep_rs[t] = running_add
        return discounted_ep_rs


# Class instance
env = MDP_env()
sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_options=N_O, n_actions=N_A, lr=LR_A)
critic = Critic(n_options=N_O, lr=LR_C, reward_decay=GAMMA)
sess.run(tf.global_variables_initializer())

# Measurement setting
reward_history = []
visits = np.zeros([N_O[0],N_O[1],N_A])
starts = np.zeros([np.sum(N_O),N_F])
ends = np.zeros([np.sum(N_O),N_F])

# Subfunctions
## Option scalar to one_hot option vector
def opt2o_h(option, layer):
    if layer == 0:
        option_vector = np.zeros(N_O[0])
    else:
        option_vector = np.zeros(N_O[1])
    option_vector[option] = 1
    return option_vector

## One_hot state vector to state scalar
def o_h2s(state):
    return np.argmax((state))

## Q_table
def output_q_tables():

    critic.q_table_o0['option1'] = critic.q_table_o0.iloc[:, 0:N_O[0]].idxmax(axis=1)
    print("")
    print(" q_table0:")
    print(critic.q_table_o0.sort_index(axis=0, ascending=True))
    critic.q_table_o0 = critic.q_table_o0.drop("option1", 1)

    critic.q_table_o1['option2'] = critic.q_table_o1.iloc[:, 0:N_O[1]].idxmax(axis=1)
    print("")
    print(" q_table1:")
    print(critic.q_table_o1.sort_index(axis=0, ascending=True))
    critic.q_table_o1 = critic.q_table_o1.drop("option2", 1)

    all_state = np.eye(N_F)
    all_new_state = []
    for i in range(N_O[1]):
        option_vector = opt2o_h(i, 1)
        all_new_state.append(np.hstack([all_state, np.tile(option_vector, (N_F, 1))]))
    all_new_state = np.vstack(all_new_state)
    all_state_name = [str([i, j]) for j in range(N_O[1]) for i in range(N_F)]
    policy_table = pd.DataFrame(index=all_state_name, columns=list(range(N_A)), dtype=np.float64)
    policy_table.iloc[:, :] = actor.sess.run(actor.acts_prob, feed_dict={actor.s: all_new_state})
    policy_table['action'] = policy_table.iloc[:, 0:N_A].idxmax(axis=1)
    print("")
    print(" policy_table:")
    print(policy_table.sort_index(axis=0, ascending=True))

# Main
for i_episode in range(MAX_EPISODE):
    if i_episode == 33:
        print("")
    s = env.reset()
    track_r = []
    done = False

    while not done:
        option_done_0 = False
        s_s = o_h2s(s)  # state scalar
        Q_0 = critic.eval_q(s_s, 0)
        actor.choose_option(0, Q_0)
        s_0 = str([s_s, actor.chosen_options[0]])  # s_0 is a string
        starts[actor.chosen_options[0], s_s] += 1

        while not option_done_0 and not done:
            option_done_1 = False
            Q_1 = critic.eval_q(s_0, 1)
            actor.choose_option(1, Q_1)
            starts[N_O[0] + actor.chosen_options[1], s_s] += 1

            while not option_done_1 and not done:
                new_s_1 = np.hstack([s, opt2o_h(actor.chosen_options[1], 1)])
                new_s_0 = np.hstack([s, opt2o_h(actor.chosen_options[0], 0)])

                a = actor.choose_action(new_s_1)

                s_, r, done = env.step(a)

                visits[actor.chosen_options[0], actor.chosen_options[1], a] += 1

                print("episode:", i_episode, "  state:", s_s, "  options", actor.chosen_options, "  action", a)

                Q_0_next = critic.eval_q(o_h2s(s_), 0)
                s_0_ = str([o_h2s(s_), actor.chosen_options[0]])
                Q_1_next = critic.eval_q(s_0_, 1)

                beta1 = actor.eval_option_term(0, Q_0_next)
                beta2 = actor.eval_option_term(1, Q_1_next)

                critic.store_transition(0, s_s, r)
                critic.store_transition(1, s_0, r)
                actor.store_transition(new_s_1, a)

                # if done:
                #     v_ = 0
                # else:
                #     U = (1 - beta1) * Q_0_next[actor.chosen_options[0]] + beta1 * np.max(Q_0_next)
                #     v_ = GAMMA * (1 - beta2) * Q_1_next[actor.chosen_options[1]] + GAMMA * beta2 * U
                #
                # td_term = Q_1_next[actor.chosen_options[1]] - np.max(Q_1_next)

                track_r.append(r)

                s = s_
                s_s = o_h2s(s)
                s_0 = str([s_s, actor.chosen_options[0]])

                if np.random.uniform() < beta2 or done:
                    option_done_1 = True
                    ends[N_O[0] + actor.chosen_options[1], s_s] += 1

                    if sum(track_r) > 0:

                        Q_1_target = 0 if done else np.max(Q_1_next)
                        error_ep = critic.learn_ep(1, Q_1_target, actor.chosen_options[1])
                        actor.learn_ep(error_ep)

            if np.random.uniform() < beta1 or done:
                option_done_0 = True
                ends[actor.chosen_options[0], s_s] += 1

                if sum(track_r) > 0:

                    Q_0_target = 0 if done else np.max(Q_0_next)
                    critic.learn_ep(0, Q_0_target, actor.chosen_options[0])
    actor.anneal(1)

    actor.anneal(0)

    ep_rs_sum = sum(track_r)

    reward_history.append(ep_rs_sum)

    print("episode:", i_episode, "  reward: %.3f" % ep_rs_sum, "  epsilon", actor.epsilon)

average_reward = np.zeros(MAX_EPISODE)
for i in range(MAX_EPISODE):
    average_reward[i] = np.sum(reward_history[:i + 1]) / (i + 1)
output_q_tables()

print("")