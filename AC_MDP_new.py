"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from MDP_env import MDP_env
from collections import defaultdict

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped
env = MDP_env()

# N_F = env.observation_space.shape[0]
# N_A = env.action_space.n
N_F = 6
N_A = 2
N_O = (2, 4)


def one_hot_0(option):
    option_vector = np.zeros(N_O[0])
    option_vector[option] = 1
    return option_vector

def one_hot_1(option):
    option_vector = np.zeros(N_O[-1])
    option_vector[option] = 1
    return option_vector


class Actor(object):
    def __init__(self, sess, n_features, n_options, n_actions, lr=0.001):
        self.sess = sess
        self.epsilon = np.ones(2)
        self.n_options = n_options
        self.chosen_options = np.zeros(len(n_options), dtype=np.int32)
        self.new_features = n_features + n_options[-1]

        self.s = tf.placeholder(tf.float32, [None, self.new_features], "state")
        # self.s = tf.placeholder(tf.float32, [1, n_features], "state")

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
            self.exp_v = tf.reduce_mean(log_prob * self.td_error[-1])  # advantage (TD_error) guided loss

        with tf.variable_scope('train_actor'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

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

    def anneal(self, l):
        if self.epsilon[l] > 0.1:
            self.epsilon[l] = self.epsilon[l] - 0.0005


class Critic(object):
    def __init__(self, sess, n_features, n_options, lr=0.01):
        self.sess = sess
        self.n_options = n_options
        self.new_features_0 = n_features + n_options[0]

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.s_0 = tf.placeholder(tf.float32, [None, self.new_features_0], "state_0")
        self.o = tf.placeholder(tf.int32, [None, len(n_options)], "chosen_options")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        # self.b1 = tf.placeholder(tf.float32, None, 'b1')
        # self.b2 = tf.placeholder(tf.float32, None, 'b2')

        with tf.variable_scope('options_indexer'):
            self.options_indexer = []
            for i in range(len(n_options)):
                begin = [0, i]
                size = [tf.shape(self.o)[0], 1]
                slice_o = tf.squeeze(tf.slice(self.o, begin, size))
                id = tf.squeeze(tf.range(tf.shape(self.o)[0], dtype=tf.int32))
                self.options_indexer.append(tf.stack([id, slice_o], axis=0))  # only when self.o.shape[0] == 1
                # self.options_indexer.append(tf.stack([id, slice_o], axis=1))


        with tf.variable_scope('Critic0'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=40,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v0 = tf.layers.dense(
                inputs=l1,
                units=n_options[0],  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V0'
            )

        with tf.variable_scope('squared_TD_error0'):
            self.chosen_0_v = tf.gather_nd(self.v0, tf.expand_dims(self.options_indexer[0], 0), "chosen_V")
            # self.chosen_0_v = tf.gather_nd(self.v0, tf.constant([[0, 0]]), "V")
            # self.td_error_0 = self.r + self.v_ - self.v0
            self.td_error_0 = self.r + self.v_ - self.chosen_0_v
            self.loss_0 = tf.square(self.td_error_0)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train_0'):
            self.train_op_0 = tf.train.AdamOptimizer(lr).minimize(self.loss_0)

        with tf.variable_scope('Critic1'):
            l1 = tf.layers.dense(
                inputs=self.s_0,
                units=40,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v1 = tf.layers.dense(
                inputs=l1,
                units=n_options[1],  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V1'
            )

        with tf.variable_scope('squared_TD_error1'):
            self.chosen_1_v = tf.gather_nd(self.v1, tf.expand_dims(self.options_indexer[1], 0), "chosen_V")
            # self.chosen_1_v = tf.gather_nd(self.v1, tf.constant([[0, 0]]), "V")
            self.td_error_1 = self.r + self.v_ - self.chosen_1_v
            self.loss_1 = tf.square(self.td_error_1)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train_1'):
            self.train_op_1 = tf.train.AdamOptimizer(lr).minimize(self.loss_1)

    def learn(self, s, s_0, r, s_, v_, o):
        s, s_0, s_ = s[np.newaxis, :], s_0[np.newaxis, :], s_[np.newaxis, :]

        o = o[np.newaxis, :]
        td_error0, _ = self.sess.run([self.td_error_0, self.train_op_0],
                                    {self.s: s, self.v_: v_, self.r: r, self.o: o})
        td_error1, _ = self.sess.run([self.td_error_1, self.train_op_1],
                                    {self.s_0: s_0, self.v_: v_, self.r: r, self.o: o})
        return td_error0 + td_error1

    def eval_q(self, state, layer, chosen_option=None):
        state = state[np.newaxis, :]
        if layer == 0:
            Q_options = self.sess.run(self.v0, {self.s: state})
        else:
            Q_options = self.sess.run(self.v1, {self.s_0: state})
        Q_options = np.ravel(Q_options)
        if chosen_option is None:
            return Q_options
        else:
            return Q_options[chosen_option]



def plot_figures():
    # plt.figure(1)
    # plt.plot(range(i_episode+1), reward_list)
    # plt.xlabel("i_episode")
    # plt.ylabel("running_reward")

    plt.figure(1)
    plt.plot(range(i_episode+1), average_reward)
    plt.xlabel("i_episode")
    plt.ylabel("average_reward")

def output_q_tables():
    # all_state = np.arange(n_states).reshape((n_states, 1))
    all_state = np.eye(N_F)
    q_table0 = pd.DataFrame(index=list(range(N_F)), columns=list(range(N_O[0])), dtype=np.float64)
    q_table0.iloc[:,:] = critic.sess.run(critic.v0, feed_dict={critic.s: all_state})
    q_table0['option1'] = q_table0.iloc[:, 0:N_O[0]].idxmax(axis=1)
    print("")
    print(" q_table0:")
    print(q_table0.sort_index(axis=0, ascending=True))

    all_new_state = []
    for i in range(N_O[0]):
        option_vector = one_hot_0(i)
        all_new_state.append(np.hstack([all_state, np.tile(option_vector, (N_F, 1))]))
    all_new_state = np.vstack(all_new_state)
    q_table1 = pd.DataFrame(index=list(range(len(all_new_state))), columns=list(range(N_O[1])), dtype=np.float64)
    q_table1.iloc[:,:] = critic.sess.run(critic.v1, feed_dict={critic.s_0: all_new_state})
    q_table1['option2'] = q_table1.iloc[:, 0:N_O[1]].idxmax(axis=1)
    print("")
    print(" q_table1:")
    print(q_table1.sort_index(axis=0, ascending=True))



sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_options=N_O, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, n_options=N_O,
                lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

reward_list = []
reward_history = []

visits = np.zeros([N_O[0],N_O[1],N_A])
starts = np.zeros([np.sum(N_O),N_F])
ends = np.zeros([np.sum(N_O),N_F])

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    done = False

    while not done and t < MAX_EP_STEPS:
        # if RENDER: env.render()
        option_done_0 = False
        Q_0 = critic.eval_q(s, 0)
        actor.choose_option(0, Q_0)
        s_0 = np.hstack([s, one_hot_0(actor.chosen_options[0])])
        starts[actor.chosen_options[0],np.argmax(s)] += 1

        while not option_done_0 and not done:
            option_done_1 = False
            Q_1 = critic.eval_q(s_0, 1)
            actor.choose_option(1, Q_1)
            starts[N_O[0]+actor.chosen_options[1], np.argmax(s)] += 1

            while not option_done_1 and not done:
                new_s = np.hstack([s, one_hot_1(actor.chosen_options[-1])])

                a = actor.choose_action(new_s)

                s_, r, done = env.step(a)

                visits[actor.chosen_options[0],actor.chosen_options[1],a] += 1

                print("episode:", i_episode, "  state:", np.argmax(s), "  options", actor.chosen_options, "  action", a)

                Q_0_next = critic.eval_q(s_, 0)

                s_0_ = np.hstack([s_, one_hot_0(actor.chosen_options[0])])
                Q_1_next = critic.eval_q(s_0_, 1)

                beta1 = actor.eval_option_term(0, Q_0_next)
                beta2 = actor.eval_option_term(1, Q_1_next)

                if done:
                    v_ = 0
                else:
                    U = (1 - beta1) * Q_0_next[actor.chosen_options[0]] + beta1 * np.max(Q_0_next)
                    v_ = GAMMA * (1 - beta2) * Q_1_next[actor.chosen_options[1]] + GAMMA * beta2 * U

                track_r.append(r)

                td_error = critic.learn(s, s_0, r, s_, v_, actor.chosen_options)  # gradient = grad[r + gamma * V(s_) - V(s)]
                actor.learn(new_s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

                s = s_
                t += 1

                if np.random.uniform() < beta2:
                    option_done_1 = True
                    ends[N_O[0]+actor.chosen_options[1], np.argmax(s)] += 1

            if np.random.uniform() < beta1:
                option_done_0 = True
                ends[actor.chosen_options[0], np.argmax(s)] += 1
    actor.anneal(1)

    actor.anneal(0)

    ep_rs_sum = sum(track_r)

    reward_history.append(ep_rs_sum)

    if 'running_reward' not in globals():
        running_reward = ep_rs_sum
        reward_list.append(running_reward)
    else:
        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
        reward_list.append(running_reward)
    # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
    print("episode:", i_episode, "  reward: %.3f" % running_reward, "  epsilon", actor.epsilon)

average_reward = np.zeros(MAX_EPISODE)
for i in range(MAX_EPISODE):
    average_reward[i] = np.sum(reward_history[:i+1]) / (i+1)

print("")


