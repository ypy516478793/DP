import random
import numpy as np

random.seed(1)

class MDP_env:
    def __init__(self):
        self.visited_six = False
        self.current_state = 1
        self.state = one_hot(self.current_state)

    def reset(self):
        self.visited_six = False
        self.current_state = 1
        self.state = one_hot(self.current_state)
        return self.state

    def step(self, action):
        if self.current_state != 0:
            # If "right" selected
            if action == 1:
                # if self.current_state < 5:  # if current_state == 5, do not change current_state
                #     self.current_state += 1
                if random.random() < 0.5:
                    if self.current_state < 5:  # if current_state == 5, do not change current_state
                        self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 5:
                self.visited_six = True

        self.state = one_hot(self.current_state)

        if self.current_state == 0:
            if self.visited_six:
                return self.state, 1.00, True
            else:
                return self.state, 1.00/100.00, True
        else:
            return self.state, 0.0, False

def one_hot(state):
    state_vector = np.zeros(6)
    state_vector[state] = 1
    return state_vector
