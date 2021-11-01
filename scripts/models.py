import numpy as np

from expansions import polynomial_expansion
from implementations import ridge_regression#, logistic_regression_penalized_gradient_descent
from proj1_helpers import predict_labels
from logistic_regression import logistic_regression_penalized_gradient_descent

def compute_weights(tX, y, models) : 
    """Computes the weights for each jet model

    Args:
        tX (list of matrices): train dataset divided by jet
        y (list of arrays): outcome divided by jet
        models (dict): models to be used for each jet

    Returns: 
        The weights to be used for each jet
    """
    weights = []

    for i, (jet, model) in enumerate(models.items()) : 
        x_expanded = polynomial_expansion(tX[i], model["degree"], mixed_columns = model["mixed"])
        print("Training with {}...".format(model['model']))
        if model["model"] == "least squares" :
            w, err = ridge_regression(y[i], x_expanded, model["lambda"])
            weights.append(w)
        elif model["model"] == "logistic regression" : 
            w = logistic_regression_penalized_gradient_descent(y[i], x_expanded, 0.01, model["lambda"], 30)
            weights.append(w)
        else : 
            raise Exception("Model not recognised")
    return weights


def compute_predictions(x_tests_jets, w_jets, models, jet_indices) : 
    """Generates the predictions given the test dataset and the weights to be used

    Args:
        tX_test (list of matrices): test dataset divided by jet
        w (list of arrays): weights to be used for each jet

    Returns:
        The list of predictions 
    """

    print("Compute predictions...")
    y_predicted = np.zeros((len(jet_indices), 1))
    y_predicted_jet = []
    for i, (jet, model) in enumerate(models.items()) :
        x_expanded_jet = polynomial_expansion(x_tests_jets[i], model["degree"], mixed_columns=model["mixed"])
        y_predicted_jet.append(predict_labels(w_jets[i], x_expanded_jet))

    y_predicted[jet_indices==0] = y_predicted_jet[0]
    y_predicted[jet_indices==1] = y_predicted_jet[1]
    y_predicted[jet_indices>=2] = y_predicted_jet[2]

    return y_predicted