import numpy as np
import tensorflow as tf
from MDP_env import MDP_env
from Option_model import OptionModel, Critic


def run_MDP(num_episodes, option_depth, env, option_controller, critic):
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            option_terminate = [False] * option_depth
            layer = 0
            hierarchical_Q = critic.eval_q(state)
            # option_0 = option_controller.choose_option(l=layer, Q=hierarchical_Q)
            option_controller.choose_option(layer, hierarchical_Q)

            while not option_terminate[layer]:
                layer = 1
                hierarchical_Q = critic.eval_q(state)
                option_controller.choose_option(layer, hierarchical_Q)

                while not option_terminate[layer]:
                    action = option_controller.choose_action(state)
                    state_next, reward, done = env.step(action)
                    critic.done = done
                    hierarchical_Q_next = critic.eval_q(state_next)
                    chosen_option_term_prob_next = option_controller.eval_option_term(layer, hierarchical_Q_next)
                    td_error = critic.learn(state, reward, state_next, hierarchical_Q_next)




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
