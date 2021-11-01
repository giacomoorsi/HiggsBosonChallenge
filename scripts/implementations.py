import numpy as np
import math

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mean square error
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - w  : Weight vector (d by 1 vector)

    Output:
    - loss : mean square loss (scalar)
    """
    e = y - np.matmul(tx, w)
    return (1/(2*len(y)))*np.matmul(e.T, e)

def mse_to_rmse(mse):
    """Return the rmse given the mse value"""
    return math.sqrt(2*mse)


def compute_loss_mae(y, tx, w):
    """Calculate the loss using mean absolute error
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - w  : Weight vector (d by 1 vector)

    Output:
    - loss : mean absolute value loss (scalar)
    """
    return sum(math.abs(y-np.matmul(tx,w)))/len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_gradient_lse(y, tx, w):
    """Compute the gradient of least squares loss function.
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - w  : Weight vector (d by 1 vector)

    Output:
    - Gradient : The gradient of least squares loss function (d by 1 vector)
    """
    e = y-np.dot(tx,w)
    return - 1/len(y) * np.dot(tx.T, e)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for least squares loss function.
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - initial_w : The initial weight vector (d by 1 vector)
    - max_iters : The number of iterations/steps to execute gradient descend (integer)
    - gamma : Learning rate (scalar)

    Outputs:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The final loss corresponds to the final weight 'w' (scalar)
    """
    # Initialize the weight to start
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_lse(y, tx ,w)
        # Update the current weight using step size 'gamma' and current gradient
        w = w - gamma * grad 

    # Compute the loss after obtain the final weight
    loss = compute_loss_mse(y, tx, w)
    
    return w, mse_to_rmse(loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for least squares loss, and batch size = 1
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - initial_w : The initial weight vector (d by 1 vector)
    - max_iters : The number of iterations/steps to execute gradient descend (integer)
    - gamma : Learning rate (scalar)

    Outputs:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The final loss corresponds to the final weight 'w' (scalar)
    """
    
    # Initialize the weight to start
    w = initial_w
    for n_iter in range(max_iters) : 
        # Batch size is always equals to 1
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1) : 
            # Compute the gradient by the given minibatch y & tx
            grad = compute_gradient_lse(minibatch_y, minibatch_tx, w)
            # Update the current weight using step size 'gamma' and current gradient
            w = w - gamma * grad 

    # Compute the loss after obtain the final weight
    loss = compute_loss_mse(y, tx, w)
    
    return w, mse_to_rmse(loss)



def least_squares(y, tx):
    """Calculate the least squares solution with the closed form linear equations.
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)

    Output:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The mean square loss for this optimal 'w'
    """
    # Solve w for equation (tx^T * tx) * w = t^T * y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_mse(y, tx, w)
    
    return w, mse_to_rmse(loss)



def ridge_regression(y, tx, lambda_) :
    """Calculate the least squares solution with a regularization given parameter lambda_
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - lambda_ : The regularization parameter

    Output:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The mean square loss for this optimal 'w'
    """
    # Get the dimension of the data matrix tx
    n, d = tx.shape
    w = np.zeros(d)
    # Solve the linear function tx^T * (tx + 2n*lambda_*I) * w = tx^T * y
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)

    loss = compute_loss_mse(y, tx, w)

    return w, mse_to_rmse(loss)


#---------------------------- #
# Logistic regression related #
#-----------------------------#


def sigmoid(t):
    """Apply the sigmoid function on t."""
    out = np.ones((len(t), 1))
    to_be_computed = np.logical_and(t < 30, t > -30)
    out[to_be_computed] = 1 / (1 + np.exp(-t[to_be_computed]))
    out[t > -30] = 0
    return out


def compute_loss_logistic(y, tx, w):
    """Compute the loss: negative log likelihood for logistic regression"""
    prod = tx @ w
    a = prod
    # Consider function a = log(1 + exp(prod)),
    #   if prod is too small, it's close to log(1) = 0
    a[prod < (-10)] = 0
    #   if prod is large, it's close to log(exp(prod)) = prod
    a[prod > 10] = prod[prod > 10]
    # The rest goes to regular computation
    incl = np.logical_and(prod >= -10, prod <= 10)
    a[incl] = np.log(1 + np.exp(prod[incl]))

    b = y * prod

    return sum(a - b)


def compute_gradient_logistic(y, tx, w):
    """Compute the gradient of logistic regression with negative log likelihood loss function."""
    return tx.T @ (sigmoid(tx @ w) - y)

def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """Compute the gradient of regularized logistic regression with negative log likelihood loss function."""
    return compute_gradient_logistic(y, tx, w) + 2*lambda_*w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for negative log likelihood loss function.
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - initial_w : The initial weight vector (d by 1 vector)
    - max_iters : The number of iterations/steps to execute gradient descend (integer)
    - gamma : Learning rate (scalar)

    Outputs:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The final loss corresponds to the final weight 'w' (scalar)
    """
    # Initialize the weight to start
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx ,w)
        # Update the current weight using step size 'gamma' and current gradient
        w = w - gamma * grad 

    # Compute the loss after obtain the final weight
    loss = compute_loss_logistic(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm for negative log likelihood loss function.
    Inputs:
    - y  : True answer (N by 1 vector)
    - tx : Data (N by d matrix)
    - lambda_   : The regularization parameter
    - initial_w : The initial weight vector (d by 1 vector)
    - max_iters : The number of iterations/steps to execute gradient descend (integer)
    - gamma : Learning rate (scalar)

    Outputs:
    - w : The weight vector after gradient descend exploration  (d by 1 vector)
    - loss : The final loss corresponds to the final weight 'w' (scalar)
    """
    # Initialize the weight to start
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_reg_logistic(y, tx ,w, lambda_)
        # Update the current weight using step size 'gamma' and current gradient
        w = w - gamma * grad 

    # Compute the loss after obtain the final weight
    loss = compute_loss_logistic(y, tx, w)

    return w, loss