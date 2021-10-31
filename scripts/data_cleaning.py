import numpy as np

JET_COLUMN = 22
JET_DISCARD_FEATURES = [[4, 5, 6, 12, JET_COLUMN, 23, 24, 25, 26, 27, 28],
                        [4, 5, 6, 12, JET_COLUMN, 26, 27, 28],
                        [JET_COLUMN]]


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
    return [y_jet0, y_jet1, y_jet2], [x_jet0, x_jet1, x_jet2]


def split_test_data(x_test):
    x_test_jet0 = x_test[x_test[:, JET_COLUMN] == 0]
    x_test_jet1 = x_test[x_test[:, JET_COLUMN] == 1]
    x_test_jet2 = x_test[x_test[:, JET_COLUMN] >= 2]
    return [x_test_jet0, x_test_jet1, x_test_jet2]


def drop_x_features(x_jet, jet_num):
    return np.delete(x_jet, JET_DISCARD_FEATURES[jet_num], axis=1)


def replace_nan_with_medians(x_jet, medians):
    # x = np.nan_to_num(x)
    for i in range(x_jet.shape[1]):
        x_jet[:, i] = np.nan_to_num(x_jet[:, i], nan=medians[i])
    return x_jet


def replace_missing_with_nan(x):
    x[x == -999] = np.NaN
    return x


def replace_outliers_with_nan(x, keep=0.95):
    """Replace outliers with NaN"""
    for i in range(x.shape[1]):
        min_value = np.quantile(x[:, i], (1 - keep) / 2)
        max_value = np.quantile(x[:, i], (1 + keep) / 2)
        values_to_be_changed = np.logical_or(x[:, i] < min_value, x[:, i] > max_value)
        x[values_to_be_changed, i] = np.nan
    return x


def remove_useless_columns(x, stds):
    """Removes columns with standard deviation == 0"""
    x = np.delete(x, np.where(stds == 0), axis=1)
    stds = stds[stds != 0]
    return x, stds


def standardize(x, means, stds):
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - means[i]) / stds[i]
    return x
