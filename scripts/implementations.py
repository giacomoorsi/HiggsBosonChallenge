import numpy as np
import math
from helper import *

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mean square error"""
    e = y - np.matmul(tx,w.T) 
    return (1/(2*len(y)))*np.matmul(e.T, e)

def compute_loss_mae(y, tx, w):
    """Calculate the loss using mean absolute error"""
    return sum(math.abs(y-np.matmul(tx,w)))/len(y)

def compute_loss(y,tx,w) : 
    return compute_loss_mse(y,tx,w)

def compute_gradient_lse(y, tx, w):
    """Compute the gradient."""
    e = y-np.dot(tx,w)
    return - 1/len(y) * np.dot(tx.T, e)

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_lse(y, tx ,w)
        # print("grad: ", grad)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad 
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        ws.append(w)
        losses.append(loss)
    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters) : 
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size) : 
            grad = compute_gradient_lse(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y, tx, w)
            w = w - gamma * grad 
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format( bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            ws.append(w)
            losses.append(loss)
    
    return losses, ws

# Least squares fitting

def least_squares(y, tx, rmse=0):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss_mse(y, tx, w)
    if rmse : 
        mse = math.sqrt(2*mse)
        return mse, w
    else :
        return mse, w



def ridge_regression(y, tx, lambda_) :
    prod = tx.T@tx
    lambda_1 = 2*len(y)*lambda_
    return np.linalg.inv(prod + lambda_1*np.eye(prod.shape[0], dtype=int))@tx.T@y