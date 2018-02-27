import numpy as np
import tensorflow as tf
from MDP_env import MDP_env
from Option_model import OptionModel, Critic


def run_MDP(num_episodes, option_depth, env, option_controller, critic):
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            layer = 0
            option_0 = critic.chosen_options[0]


def train(args):

    num_episodes = args.NUM_EPISODES
    option_depth = len(args.NUM_OPTIONS)
    output_graph = args.OUTPUT_GRAPH


    sess = tf.Session()
    env = MDP_env()

    option_controller = OptionModel(sess=sess,
                                    n_features=args.NUM_FEATURES,
                                    n_actions=args.NUM_ACTIONS,
                                    n_options=args.NUM_OPTIONS,
                                    lr=args.LR_OPTION,
                                    e_greedy_min=args.EPSILON_MIN,
                                    e_decrement=(1-args.EPSILON_MIN)/num_episodes)

    critic = Critic(sess=sess,
                    n_features=args.NUM_FEATURES,
                    n_options=args.NUM_OPTIONS,
                    lr=args.LR_CRITIC,
                    gamma=args.REWARD_DISCOUNT,)

    sess.run(tf.global_variables_initializer())

    if output_graph:
        tf.summary.FileWriter("logs/", sess.graph)

    run_MDP(num_episodes, option_depth, env, option_controller, critic)
