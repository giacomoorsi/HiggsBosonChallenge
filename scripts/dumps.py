def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    # a = y*np.log(sigmoid(tx@w))
    # b = (1-y)*np.log(1-sigmoid(tx@w))
    # c = -np.sum(a+b)
    prod = tx @ w
    a = prod
    a[prod < (-10)] = 0
    a[prod > 10] = prod[prod > 10]
    incl = np.logical_and(prod <= -10, prod <= 10)
    a[incl] = np.log(1 + np.exp(prod[incl]))
    b = y * prod
    return sum(a - b)

def feature_expansion(x, degree) :
    x_expanded = np.ones((len(x), 1))

    for d in range(0, degree+1) :
        x_elevated = np.power(x,d)
        x_expanded = np.c_[x_expanded, x_elevated]


    # x_expanded = np.c_[x_expanded, np.sqrt(np.abs(x))]
    # cross terms
    # vectorize the calculation
    # ref: https://stackoverflow.com/questions/22041561/python-all-possible-products-between-columns
    #i, j = np.triu_indices(x.shape[1], 1)
    #x_expanded = np.c_[x_expanded, x[:, i] * x[:, j]]


    return x_expanded



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    n_samples_training = math.ceil(ratio * x.shape[0])
    a = np.array(np.c_[x, y])
    np.random.shuffle(a)
    training, test = a[:n_samples_training, :], a[n_samples_training:, :]
    training_x, training_y = training[:, 0], training[:, 1]
    test_x, test_y = test[:, 0], test[:, 1]
    return training_x.T, training_y.T, test_x.T, test_y.T


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
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent", True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))

def logistic_regression_penalized_gradient_descent(y, tx, gamma, lambda_, max_iter):
    threshold = 1e-8
    losses = []
    w = np.zeros((tx.shape[1], 1))

    for i in range(max_iter):
        # get loss and update w.
        #   print("Current iteration={i}".format(i=i))
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #   if i % 100 == 0:
        #      print("Current iteration={i}, loss={l}".format(i=i, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w

def logistic_function(y, tx, w):
    yXw = y * (tx @ w)

    # l = np.sum(np.log(1. + np.exp(-yXw)))
    l = calculate_loss(y, tx, w)
    # g = tx.T @ (- y / (1. + np.exp(yXw)))
    g = calculate_gradient(y, tx, w)
    return l, g

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
