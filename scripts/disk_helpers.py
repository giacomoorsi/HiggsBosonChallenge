import time
import os
import numpy as np

PATH = '../data/accuracies/'


def save_data(training_method, jet_string, degrees, lambdas, results):
    """
    Save the training data for later use.
    :param training_method: name of training method used.
    :param jet_string: name of subset used.
    :param degrees: array of degrees used.
    :param lambdas: array of lambdas used.
    :param results: matrix of accuracies, degrees as rows and lambdas as columns
    :return: None
    """
    extended_path = "{}{}-{}-{}/".format(PATH, time.strftime("%Y%m%d-%H%M%S"), jet_string, training_method)
    os.mkdir(extended_path)
    np.save(extended_path + 'degrees', degrees)
    np.save(extended_path + 'lambdas', lambdas)
    np.save(extended_path + 'results', results)


def load_np_array(folder_name, selection):
    """
    Load the degrees array used for training.

    :param selection: specify with 'd', 'l', or 'r' if you want to load the degrees/lambdas/results arrays.
    :param folder_name: specify the training session from which to recover information.
    :return: one-dimensional numpy array.
    """
    if selection == 'd':
        file_name = 'degrees.npy'
    elif selection == 'l':
        file_name = 'lambdas.npy'
    elif selection == 'r':
        file_name = 'results.npy'
    else:
        print("You provided an invalid selection.")
        return
    file_path = '{}{}/{}'.format(PATH, folder_name, file_name)
    return np.load(file_path)
