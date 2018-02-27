import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from Option_model import OptionModel, Critic
from MDP_env import MDP_env
from collections import defaultdict

def eval_Q(state,  option_controller, critic, chosen_options=None):
    Q_options = critic.eval_q(state)
    hierarchical_q = option_controller.reshape(Q_options)
    if chosen_options is None:
        return hierarchical_q
    else:
        chosen_options_q = []
        for i in range(len(chosen_options)):
            chosen_options_q[i] = hierarchical_q[i][chosen_options[i]]
        return chosen_options_q


def eval_options_term(options_prob, option_depth, chosen_options=None):
    options_term_prob = defaultdict(list)
    chosen_options_term_prob = []
    for i in range(option_depth-1):
        options_term_prob[i] = 1 - options_prob[i]
        chosen_options_term_prob[i] = 1 - options_prob[i][chosen_options[i]]
    if chosen_options is None:
        return options_term_prob
    else:
        return chosen_options_term_prob


def eval_term_prob(state, critic, chosen_options):
    terminate_prob = critic.eval_t(state)[chosen_options[-1]]
    return terminate_prob






def run_MDP(num_episodes, option_depth, env, option_controller, critic):


    # total_step = 0  # number of total transitions
    for i_episode in range(num_episodes):

        # initial state
        state = env.reset()

        done = False
        while not done:
            chosen_options = []
            option_terminate = [False] * option_depth
            layer = 0
            # choose options in layer[0]
            hierarchical_Q = eval_Q(state, option_controller, critic)
            options_prob = option_controller.options_prob(hierarchical_Q)
            chosen_options[layer] = np.random.choice(np.arange(options_prob[layer].size), p=options_prob[layer])

            while not option_terminate[layer]:
                # choose options in layer[1]
                layer = 1
                hierarchical_Q = eval_Q(state, option_controller, critic)
                options_prob = option_controller.options_prob(hierarchical_Q)
                chosen_options[layer] = np.random.choice(np.arange(options_prob[layer].size), p=options_prob[layer])

                while not option_terminate[layer]:
                    # choose an action
                    action = option_controller.choose_action(state)
                    # execute action and obtain next state and extrinsic reward from environment
                    state_next, reward, done = env.step(action)
                    # Compute option termination probability
                    hierarchical_Q_next = eval_Q(state_next, option_controller, critic)
                    options_prob_next = option_controller.options_prob(hierarchical_Q_next)
                    chosen_options_term_prob_next = eval_options_term(options_prob_next, option_depth, chosen_options)
                    # Option evaluation
                    chosen_options_Q = eval_Q(state, option_controller, critic, chosen_options)
                    deltas = reward * np.ones(option_depth) - chosen_options_Q
                    if not done:
                        terminate_prob_next = eval_term_prob(state_next, critic, chosen_options)
                        chosen_options_Q_next = eval_Q(state_next, option_controller, critic, chosen_options)
                        options_Q_next = eval_Q(state_next, option_controller, critic)[0]
                        U = (1 - chosen_options_term_prob_next[0]) * chosen_options_Q_next[0] + \
                            chosen_options_term_prob_next[0] * np.argmax(options_Q_next)

                        deltas += critic.gamma * (1 - terminate_prob_next) * chosen_options_Q_next[-1] +\
                                  critic.gamma * terminate_prob_next * U











            goal = meta_controller.choose_action(state_array)
            # goal = 2          ###############
            controller.goal_attempts[goal] += 1
            done = False

            while not done:
                extrinsic_reward = 0
                start_state = state_array
                step = 0
                goal_reached = False

                while not done and not goal_reached:
                    # while s is not terminal or goal is not reached
                    action = controller.choose_action(state_array, goal=goal)
                    # execute action and obtain next state and extrinsic reward from environment
                    state_, ex_reward, done = env.step(action)
                    visits[id_episode][state_] += 1
                    # obtain intrinsic reward from internal critic
                    in_reward, goal_reached = meta_controller.criticize(state_, goal)
                    state__array = one_hot(state_)
                    # store transition ({s, g}, a, r, {s_, g}) for controller
                    controller.store_transition(state_array, action, in_reward, state__array, goal)
                    if total_step > 200:
                        # update parameters for controller
                        controller.learn()
                        # update parameters for meta_controller
                        meta_controller.learn()

                    print("Episode: " + str(i_episode + 1),
                          " Steps: " + str(step + 1),
                          " Total_step: " + str(total_step + 1),
                          " (State, Action): " + str((state, action)),
                          " State_: " + str(state_),
                          " Goal: " + str(goal),
                          " meta_e: " + str(meta_controller.epsilon[0]),
                          " con_e: " + str(controller.epsilon)
                          )

                    extrinsic_reward += ex_reward
                    state = state_
                    state_array = one_hot(state)
                    # state_array = np.array([state])
                    step += 1
                    total_step += 1

                if goal_reached:
                    controller.goal_success[goal] += 1
                    print("Goal reached!")

                # # update parameters for meta_controller
                # meta_controller.learn()
                # store transition (s_0, g, ex_r, s_) for meta_controller
                meta_controller.store_transition(start_state, goal, extrinsic_reward, state__array)
                # anneal epsilon greedy rate for controller
                controller.anneal(step, goal=goal, adaptively=False, success=goal_reached)
                # # anneal epsilon greedy rate for meta_controller
                # meta_controller.anneal()

                if not done:  # when goal is terminal, goal_reached = True & done = True
                    # choose a new goal
                    goal = meta_controller.choose_action(state_array)
                    # goal = 2          ###############
                    controller.goal_attempts[goal] += 1

                total_extrinsic_reward[i_episode] = total_extrinsic_reward[i_episode - 1] + extrinsic_reward

        # anneal epsilon greedy rate for meta_controller
        meta_controller.anneal()


        if (i_episode + 1) % (num_episodes / period) == 0:
            id_period = i_episode // (num_episodes // period)
            for k in range(n_goals):
                goal_attempts[id_period, k] = controller.goal_attempts[k] - \
                                              np.sum(goal_attempts[:id_period, k])
                goal_success[id_period, k] = controller.goal_success[k] - \
                                             np.sum(goal_success[:id_period, k])


def one_hot(state):
    state_vector = np.zeros([n_features])
    state_vector[state] = 1
    return state_vector





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



    # output_q_tables()
    # plot_figures()


if __name__ == "__main__":
    pass

