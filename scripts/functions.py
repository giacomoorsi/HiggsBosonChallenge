import numpy as np

def feature_expansion(x, degree) : 
    x_expanded = np.ones((len(x), 1))
    
    for d in range(1, degree+1) : 
        x_elevated = np.power(x,d)
        x_expanded = np.c_[x_expanded, x_elevated]
    
    
    x_expanded = np.c_[x_expanded, np.sqrt(np.abs(x))]
    # cross terms
    # vectorize the calculation
    # ref: https://stackoverflow.com/questions/22041561/python-all-possible-products-between-columns
    i, j = np.triu_indices(x.shape[1], 1)
    x_expanded = np.c_[x_expanded, x[:, i] * x[:, j]]
    
    
    return x_expanded


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    n_samples_training = math.ceil(ratio*x.shape[0])
    a = np.array(np.c_[x,y])
    np.random.shuffle(a)
    training, test = a[:n_samples_training,:], a[n_samples_training:,:]
    training_x, training_y = training[:, 0], training[:, 1]
    test_x, test_y = test[:, 0], test[:, 1]
    return training_x.T, training_y.T, test_x.T, test_y.T