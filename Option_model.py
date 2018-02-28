"""
Option-Critic using TD-error as the Advantage, Reinforcement Learning.

"""

import numpy as np
import tensorflow as tf
from collections import defaultdict

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class OptionModel(object):
    def __init__(self, sess, n_features, n_options, n_actions, lr=0.001, e_greedy_min=0.05, e_decrement=0.01):
        self.sess = sess
        self.n_options = n_options
        self.depth = len(n_options)
        self.epsilon = np.ones(len(n_options))
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

    def eval_options_probs(self, Q):
        options_probs = defaultdict(list)
        for i in range(self.depth-1):
            options_probs[i] = np.ones(self.n_options[i]) * self.epsilon[i] / self.n_options[i]
            options_probs[i][np.argmax(Q[i])] += 1 - self.epsilon[i]
        return  options_probs

    # def choose_option(self, l, Q):
    #     if np.random.uniform() > self.epsilon[l]:
    #         option = np.argmax(Q[l])
    #     else:
    #         option = np.random.randint(0, self.n_options)
    #     return option

    def choose_option(self, l, Q):
        options_probs = self.eval_options_probs(Q)
        self.chosen_options[l] = np.random.choice(np.arange(options_probs[l].size), p=options_probs[l])
        # return self.chosen_options[l]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def eval_option_term(self, l, Q_):
        options_probs_next = self.eval_options_probs(Q_)
        options_term_prob_next = 1 - options_probs_next[l-1]
        return options_term_prob_next[self.chosen_options[l-1]]





class Critic(object):
    def __init__(self, sess, n_features, n_options, lr=0.01, gamma=0.99):
        self.sess = sess
        self.gamma = gamma
        self.n_options = n_options
        self.depth = len(n_options)
        self.done = False

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, sum(n_options)], "v_next")
        self.o = tf.placeholder(tf.int32, [None, len(n_options)], "chosen_options")
        self.r = tf.placeholder(tf.float32, [None, len(n_options)], "reward")
        self.b = tf.placeholder(tf.float32, [None, len(n_options)-1], "option_term")

        options_slice_indexer = []
        for i in range(len(n_options)):
            begin_point = [0, i]
            end_point = [tf.shape(self.o)[0], i + 1]
            slice_o = tf.squeeze(tf.slice(self.o, begin_point, end_point))
            options_slice_indexer.append(tf.stack([tf.range(tf.shape(self.o)[0], dtype=tf.int32), slice_o], axis=1))
        self.options_indexer = tf.stack(options_slice_indexer, axis=1)

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

            self.chosen_v = tf.gather_nd(self.v, self.options_indexer, "chosen_V")

            disconnected_l1 = tf.stop_gradient(l1)

            self.t = tf.layers.dense(
                inputs=disconnected_l1,
                units=n_options[-1],  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Terminate'
            )

            self.disconnected_t = tf.stop_gradient(self.t, "disconnected_term_prob")




        with tf.variable_scope('squared_TD_error'):
            if self.done:
                self.td_error = self.r - self.chosen_v
            else:
                self.td_error = self.r

            self.loss = tf.reduce_sum(0.5 * tf.square(self.td_error))

            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, beta, h_q):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

    def eval_t(self, s):
        return self.sess.run(self.disconnected_t, {self.s: s})

    def eval_q(self, state, chosen_options=None):
        state = state[np.newaxis, :]
        Q_options = self.sess.run(self.v, {self.s: state})
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
        for i in range(self.depth):
            hierarchical_q[i] = Q[pointer: pointer+self.n_options[i]]
            pointer += self.n_options[i]
        return hierarchical_q




