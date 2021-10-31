import numpy as np

JET_COLUMN = 22
JET_DISCARD_FEATURES = [[4, 5, 6, 12, JET_COLUMN, 23, 24, 25, 26, 27, 28],
                        [4, 5, 6, 12, JET_COLUMN, 26, 27, 28],
                        [JET_COLUMN]]


def clean_data(y, x, x_test):
    """
    Split the given datasets into 3 groups, depending on the corresponding jet value, and
    replace missing values

    :param y: np array containing the class labels used in training.
    :param x: np array containing observations used in training.
    :param x_test: np array containing observations used in testing.
    :return: Three triples, corresponding to the input data, split depending on jet value
    """
    x_data_sets, y_data_sets = split_train_data(y, x)  # Delete useless features for each x dataset
    x_test_data_sets = split_test_data(x_test)
    drop_x_features(x_data_sets, x_test_data_sets)
    replace_missing_mass_values(x_data_sets, y_data_sets, x_test_data_sets)
    return x_data_sets, y_data_sets, x_test_data_sets


def split_train_data(y, x):
    # Create three masks, one for each of the jet categories
    jet0_mask = x[:, JET_COLUMN] == 0
    jet1_mask = x[:, JET_COLUMN] == 1
    jet2_mask = x[:, JET_COLUMN] >= 2
    # Split y vector
    y_jet0 = y[jet0_mask]
    y_jet1 = y[jet1_mask]
    y_jet2 = y[jet2_mask]
    # Reshape arrays into column vectors
    y_jet0 = y_jet0.reshape((len(y_jet0), 1))
    y_jet1 = y_jet1.reshape((len(y_jet1), 1))
    y_jet2 = y_jet2.reshape((len(y_jet2), 1))
    # Split x matrix
    x_jet0 = x[jet0_mask]
    x_jet1 = x[jet1_mask]
    x_jet2 = x[jet2_mask]
    return [x_jet0, x_jet1, x_jet2], [y_jet0, y_jet1, y_jet2]


def split_test_data(x_test):
    x_test_jet0 = x_test[x_test[:, JET_COLUMN] == 0]
    x_test_jet1 = x_test[x_test[:, JET_COLUMN] == 1]
    x_test_jet2 = x_test[x_test[:, JET_COLUMN] >= 2]
    return [x_test_jet0, x_test_jet1, x_test_jet2]


def drop_x_features(x_data_sets, x_test_data_sets):
    for x_set, x_test_set, discard_features, i in zip(x_data_sets, x_test_data_sets, JET_DISCARD_FEATURES, range(3)):
        x_data_sets[i] = np.delete(x_set, discard_features, axis=1)
        x_test_data_sets[i] = np.delete(x_test_set, discard_features, axis=1)


def replace_missing_mass_values(x_sets, y_sets, x_test_sets):
    # Loop through jet0, 1 and 2 for all computations to have
    # more accurate replacement values instead of taking a global median
    for x_set, y_set, x_test_set in zip(x_sets, y_sets, x_test_sets):
        # Create two boolean masks, the first one keeps 'signal' rows,
        # the second one keeps rows that have a value different than -999 as mass,
        # which means the mass has been measured
        signal_mask = y_set[:, 0] == 1
        mass_present_mask = x_set[:, 0] != -999
        # Combine masks
        signal_and_mass_present_mask = np.logical_and(signal_mask, mass_present_mask)
        background_and_mass_present_mask = np.logical_and(~signal_mask, mass_present_mask)
        signal_and_mass_missing_mask = np.logical_and(signal_mask, ~mass_present_mask)
        background_and_mass_missing_mask = np.logical_and(~signal_mask, ~mass_present_mask)
        # Compute the mass' median of 'signal' and 'background' rows, dropping -999 values
        # from the computation
        signal_mass_median = np.median(x_set[signal_and_mass_present_mask, 0])
        background_mass_median = np.median(x_set[background_and_mass_present_mask, 0])
        # Replace missing values with computed medians
        x_set[signal_and_mass_missing_mask, 0] = signal_mass_median
        x_set[background_and_mass_missing_mask, 0] = background_mass_median
        # Now that the training set is cleaned, apply the mass estimates to the missing values
        # of the test set. Not having information about 's' and 'b' in the test set, an
        # average mass will be used as replacement for missing values
        mass_average = (signal_mass_median + background_mass_median) / 2
        x_test_set[x_test_set[:, 0] == -999, 0] = mass_average
