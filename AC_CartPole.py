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
import gym
from collections import defaultdict

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n
N_O = (4, 8)


class Actor(object):
    def __init__(self, sess, n_features, n_options, n_actions, lr=0.001):
        self.sess = sess
        self.epsilon = np.ones(2)
        self.n_options = n_options
        self.chosen_options = np.zeros(len(n_options), dtype=np.int32)

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")

        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error



        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
    
    def eval_options_probs(self, Q):
        options_probs = defaultdict(list)
        for i in range(2):
            options_probs[i] = np.ones(self.n_options[i]) * self.epsilon[i] / self.n_options[i]
            options_probs[i][np.argmax(Q[i])] += 1 - self.epsilon[i]
        return  options_probs

    def choose_option(self, l, Q):
        options_probs = self.eval_options_probs(Q)
        self.chosen_options[l] = np.random.choice(np.arange(options_probs[l].size), p=options_probs[l])
        # return self.chosen_options[l]

    def eval_option_term(self, l, Q_):
        options_probs_next = self.eval_options_probs(Q_)
        options_term_prob_next = 1 - options_probs_next[l-1]
        return options_term_prob_next[self.chosen_options[l-1]]

    def anneal(self, l):
        if self.epsilon[l] > 0.1:
            self.epsilon[l] = self.epsilon[l] - 0.005


class Critic(object):
    def __init__(self, sess, n_features, n_options, lr=0.01):
        self.sess = sess
        self.n_options = n_options

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.o = tf.placeholder(tf.int32, [None, len(n_options)], "chosen_options")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        # self.b1 = tf.placeholder(tf.float32, None, 'b1')
        # self.b2 = tf.placeholder(tf.float32, None, 'b2')

        options_slice_indexer = []
        for i in range(len(n_options)):
            begin = [0, i]
            size = [tf.shape(self.o)[0], 1]
            slice_o = tf.squeeze(tf.slice(self.o, begin, size))
            id = tf.squeeze(tf.range(tf.shape(self.o)[0], dtype=tf.int32))
            options_slice_indexer.append(tf.stack([id, slice_o], axis=0))  # only when self.o.shape[0] == 1
            # options_slice_indexer.append(tf.stack([id, slice_o], axis=1))
        self.options_indexer = tf.stack(options_slice_indexer, axis=0)  # only when self.o.shape[0] == 1
        # self.options_indexer = tf.stack(options_slice_indexer, axis=1)

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=sum(n_options),  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.chosen_v = tf.gather_nd(self.v, self.options_indexer, "chosen_V")
            self.td_error = self.r + self.v_ - self.chosen_v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, v_, o):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        o = o[np.newaxis, :]
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.o: o})
        return td_error
    
    def eval_q(self, state, chosen_options=None):
        state= state[np.newaxis, :]
        Q_options = self.sess.run(self.v, {self.s: state})
        Q_options = np.ravel(Q_options)
        hierarchical_q = self.reshape(Q_options)
        if chosen_options is None:
            return hierarchical_q
        else:
            chosen_options_q = []
            for i in range(len(chosen_options)):
                chosen_options_q.append(hierarchical_q[i][chosen_options[i]])
            return chosen_options_q

    def reshape(self, Q):
        hierarchical_q = defaultdict(list)
        pointer = 0
        for i in range(2):
            hierarchical_q[i] = Q[pointer: pointer+self.n_options[i]]
            pointer += self.n_options[i]
        return hierarchical_q

def plot_figures():
    plt.figure(1)
    plt.plot(range(i_episode), reward_list)
    plt.xlabel("i_episode")
    plt.ylabel("running_reward")



sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_options=N_O, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, n_options=N_O, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

reward_list = []

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    done = False

    while not done and t < MAX_EP_STEPS:
        if RENDER: env.render()
        option_done_0 = False
        hierarchical_Q = critic.eval_q(s)
        actor.choose_option(0, hierarchical_Q)

        while not option_done_0 and not done:
            option_done_1 = False
            hierarchical_Q = critic.eval_q(s)
            actor.choose_option(1, hierarchical_Q)

            while not option_done_1 and not done:



                a = actor.choose_action(s)

                s_, r, done, info = env.step(a)

                hierarchical_Q_next = critic.eval_q(s_)

                beta1 = actor.eval_option_term(1, hierarchical_Q_next)
                beta2 = actor.eval_option_term(2, hierarchical_Q_next)

                if done:
                    v_ = 0
                else:
                    U = (1 - beta1) * hierarchical_Q_next[0][actor.chosen_options[0]] + beta1 * np.max(hierarchical_Q_next[0])
                    v_ = GAMMA * (1 - beta2) * hierarchical_Q_next[1][actor.chosen_options[1]] + GAMMA * beta2 * U

                if done: r = -20

                track_r.append(r)

                td_error = critic.learn(s, r, s_, v_, actor.chosen_options)  # gradient = grad[r + gamma * V(s_) - V(s)]
                actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

                s = s_
                t += 1

                if np.random.uniform() < beta2:
                    option_done_1 = True

            if np.random.uniform() < beta1:
                option_done_0 = True
    actor.anneal(1)

    actor.anneal(0)










    ep_rs_sum = sum(track_r)



    if 'running_reward' not in globals():
        running_reward = ep_rs_sum
        reward_list.append(running_reward)
    else:
        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
        reward_list.append(running_reward)
    if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
    print("episode:", i_episode, "  reward:", int(running_reward), "  epsilon", actor.epsilon)


