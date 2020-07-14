import numpy as np


def get_step_annealing(cycle_size: int, start: int, stop: int, n_steps: int):
    """
    This function return the  step annealing function for target compression.
    :param cycle_size: integer that defies the cycle size
    :param start: start value of the annealing function
    :param stop: stop value of the annealing function
    :param n_steps: number of steps
    :return: a function which get an index and return a floating temperature value
    """
    cr_array_act = np.linspace(start, stop, n_steps)

    def func(i):
        if i < 0:
            return 1.0
        i = int(i / cycle_size)
        if i >= len(cr_array_act):
            return cr_array_act[-1]
        return cr_array_act[int(i)]

    return func


def get_exp_cycle_annealing(cycle_size_iter: int, temp_step: float, n: float):
    """
    This function return the  exp annealing function for the gumbel softmax.
    :param cycle_size_iter: integer that defies the cycle size
    :param temp_step: the step size coefficient
    :param n: a float scaling of the iteration index
    :return: a function which get an index and return a floating temperature value
    """

    def temp_func(i):
        if i < 0:
            return 1.0
        i = i % cycle_size_iter
        return np.maximum(0.5, 1 * np.exp(-temp_step * np.round(i / n)))

    return temp_func
