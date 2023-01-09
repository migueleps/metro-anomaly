import numpy as np


class EarlyStopping:

    def __init__(self, successive_iters, delta_worse, delta_better):

        self.successive_iters = successive_iters
        self.delta_getting_worse = delta_worse
        self.delta_getting_better = delta_better
        self.counter_increase = 0
        self.counter_decrease = 0
        self.min_so_far = np.inf

    def stopping_condition(self, validation_loss):

        if validation_loss < self.min_so_far:
            self.counter_increase = 0
            if np.abs(validation_loss - self.min_so_far) < self.delta_getting_better:
                self.counter_decrease += 1
            else:
                self.min_so_far = validation_loss
                self.counter_decrease = 0
        elif np.abs(validation_loss - self.min_so_far) < self.delta_getting_worse:
            self.counter_increase += 1

        return self.counter_increase >= self.successive_iters or self.counter_decrease >= self.successive_iters
