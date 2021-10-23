# -*- coding: utf-8 -*-
"""Logistic regression functions"""

import numpy as np
import math
from logistic_regression_helpers import *
from cross_validation import build_k_indices
from proj1_helpers import predict_labels

def sigmoid(t):
    """apply the sigmoid function on t."""
    out = np.ones((len(t),1))
    toBeComputed = np.logical_and(t<30, t>-30)
    out[toBeComputed] = 1 / (1+np.exp(-t[toBeComputed]))
    out[t>-30] = 0
    return out

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    #a = y*np.log(sigmoid(tx@w))
    #b = (1-y)*np.log(1-sigmoid(tx@w))
    #c = -np.sum(a+b)
    prod = tx@w
    a = prod
    a[prod<(-10)] = 0
    a[prod>10] = prod[prod>10]
    incl = np.logical_and(prod<=-10, prod<=10)
    a[incl] = np.log(1+np.exp(prod[incl]))
    b = y*prod
    return sum(a-b)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    l,g = logistic_function(y,tx,w)
    w = w - gamma*g
    return l, w

def logistic_function(y, tx, w) : 
    yXw = y * (tx@w)

    #l = np.sum(np.log(1. + np.exp(-yXw)))
    l = calculate_loss(y, tx, w)
    #g = tx.T @ (- y / (1. + np.exp(yXw)))
    g = calculate_gradient(y, tx, w)
    return l,g


from logistic_regression_helpers import de_standardize

def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent", True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    #loss = calculate_loss(y, tx, w) + lambda_*math.pow(np.linalg.norm(w),2)
    #grad = calculate_gradient(y, tx, w) + 2*lambda_*w
    l,g = logistic_function(y,tx,w)
    l+=lambda_ * math.pow(np.linalg.norm(w), 2)
    g+=2*lambda_*w
    return l, g


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    
    w = w - gamma*grad
    return loss, w


def logistic_regression_penalized_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    gamma = 0.01
    lambda_ = 0.1
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))



def logistic_regression_penalized_gradient_descent(y, tx, gamma, lambda_, max_iter):
    threshold = 1e-8
    losses = []

    w = np.zeros((tx.shape[1], 1))

    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w

def cross_validate_logistic_regression(y, tx, max_iter, gamma, lambda_, k_fold) : 
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    classifications = []

    for k in range(k_fold) : 
        test_indices = np.zeros(len(y)).astype(bool)
        test_indices[k_indices[k]] = True
        train_indices = (~test_indices).tolist()
        test_indices = test_indices.tolist()
            
        tx_train = tx[train_indices, :]
        y_train = y[train_indices]
        
        tx_test = tx[test_indices, :]
        y_test = y[test_indices]
        
        w = logistic_regression_penalized_gradient_descent(y_train, tx_train, gamma, lambda_, max_iter)
        classified = sum(predict_labels(w, tx_test)==y_test)/len(y_test)
        classifications.append(classified)

    return np.mean(classifications)