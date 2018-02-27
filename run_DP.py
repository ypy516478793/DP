"""
Run deep_policy based on option_critic methods

"""

import trainer

class SuperParameters(object):

    OUTPUT_GRAPH = False
    NUM_EPISODES = 3000
    # DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    # MAX_EP_STEPS = 1000  # maximum time step in one episode
    # RENDER = False  # rendering wastes time
    REWARD_DISCOUNT = 0.9  # reward discount in TD error
    LR_OPTION = 0.001  # learning rate for option_model
    LR_CRITIC = 0.01  # learning rate for critic
    NUM_OPTIONS = (4, 8)  # number of options
    NUM_FEATURES = 6  # number of features
    NUM_ACTIONS = 2  # number of actions
    EPSILON_MIN = 0.1  # minimum epsilon for exploration



if __name__ == '__main__':
    trainer.train(SuperParameters)