import numpy as np


def polynomial_expansion(x, degree, mixed_columns=False):
    """
    Expand the given vector to a polynomial of degree-th degree
    :param x: vector that will get expanded
    :param degree: number of newly added polynomial degrees
    :return: result of the expansion
    """
    x_expanded = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        x_expanded = np.c_[x_expanded, np.power(x, d)]

    if mixed_columns : 
        n = x.shape[1]
        r,c = np.triu_indices(n,1)
        out0 = x[:,r] * x[:,c]
        out = np.concatenate(( x, out0), axis=1)
        x_expanded = np.c_[x_expanded, out]

    return x_expanded
