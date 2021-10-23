import numpy as np


def polynomial_expansion(x, degree):
    """
    Expand the given vector to a polynomial of degree-th degree
    :param x: vector that will get expanded
    :param degree: number of newly added polynomial degrees
    :return: result of the expansion
    """
    x_expanded = np.ones((len(x), 1))
    for d in range(0, degree + 1):
        x_elevated = np.power(x, d)
        x_expanded = np.c_[x_expanded, x_elevated]
    return x_expanded
