# -*- coding: utf-8 -*-
"""Logistic regression functions"""

import math
from logistic_regression_helpers import *
from cross_validation import build_k_indices
from proj1_helpers import predict_labels


def sigmoid(t):
    """apply the sigmoid function on t."""
    out = np.ones((len(t), 1))
    to_be_computed = np.logical_and(t < 30, t > -30)
    out[to_be_computed] = 1 / (1 + np.exp(-t[to_be_computed]))
    out[t > -30] = 0
    return out


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    prod = tx @ w
    a = prod
    a[prod < (-10)] = 0
    a[prod > 10] = prod[prod > 10]
    incl = np.logical_and(prod <= -10, prod <= 10)
    a[incl] = np.log(1 + np.exp(prod[incl]))
    b = y * prod
    return sum(a - b)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss(y, tx, w) + lambda_ * math.pow(np.linalg.norm(w), 2)
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    updated_w = w - gamma * grad
    return loss, updated_w


def logistic_regression_penalized_gradient_descent(y, tx, gamma, lambda_, max_iter):
    threshold = 1e+0
    losses = []
    w = np.zeros((tx.shape[1], 1))

    for i in range(max_iter):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w


def cross_validate_logistic_regression(y, tx, max_iter, gamma, lambda_, k_fold):
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    classifications = []

    for k in range(k_fold):
        test_indices = np.zeros(len(y)).astype(bool)
        test_indices[k_indices[k]] = True
        train_indices = (~test_indices).tolist()
        test_indices = test_indices.tolist()

        tx_train = tx[train_indices, :]
        y_train = y[train_indices]

        tx_test = tx[test_indices, :]
        y_test = y[test_indices]

        w = logistic_regression_penalized_gradient_descent(y_train, tx_train, gamma, lambda_, max_iter)
        classified = sum(predict_labels(w, tx_test) == y_test) / len(y_test)
        classifications.append(classified)

    return np.mean(classifications)
