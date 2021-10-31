import numpy as np

JET_COLUMN = 22
JET_DISCARD_FEATURES = [[4, 5, 6, 12, JET_COLUMN, 23, 24, 25, 26, 27, 28],
                        [4, 5, 6, 12, JET_COLUMN, 26, 27, 28],
                        [JET_COLUMN]]


def split_train_data(y, x):
    """
    Split the given datasets into three jet number subsets.

    :param y: y train vector.
    :param x: x train matrix.
    :return: two 3-element arrays corresponding to y and x train split into jet numbers
    """
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
    print("tX and y matrices split into 3 sub-matrices, depending on jet number.")
    return [y_jet0, y_jet1, y_jet2], [x_jet0, x_jet1, x_jet2]


def split_test_data(x_test):
    """
    Split the given dataset into three jet number subsets.

    :param x_test: x test matrix.
    :return: array containing 3 matrices, each associated to a jet number.
    """
    x_test_jet0 = x_test[x_test[:, JET_COLUMN] == 0]
    x_test_jet1 = x_test[x_test[:, JET_COLUMN] == 1]
    x_test_jet2 = x_test[x_test[:, JET_COLUMN] >= 2]
    return [x_test_jet0, x_test_jet1, x_test_jet2]


def drop_x_features(x_jet, jet_num):
    """
    Drop useless columns of the passed x matrix, depending on the jet number.

    :param x_jet: x matrix unfiltered from meaningless columns.
    :param jet_num: jet number identifier.
    :return: x matrix filtered from meaningless columns.
    """
    return np.delete(x_jet, JET_DISCARD_FEATURES[jet_num], axis=1)


def replace_nan_with_medians(x_jet, medians):
    """
    Replace NaN values with median values.

    :param x_jet: x matrix with missing values marked as NaN.
    :param medians: array of medians, one for each feature, which will be injected in the missing value positions.
    :return: x matrix filled.
    """
    # x = np.nan_to_num(x)
    for i in range(x_jet.shape[1]):
        x_jet[:, i] = np.nan_to_num(x_jet[:, i], nan=medians[i])
    return x_jet


def replace_missing_with_nan(x):
    """
    Replace cells marked with -999 with NaN.

    :param x: Raw x matrix, still containing -999 values marking unmeasured features from observations.
    :return: Matrix with missing values marked as NaN
    """
    x[x == -999] = np.NaN
    return x


def replace_outliers_with_nan(x, keep=0.95):
    """
    Replace selected fraction of outliers with NaN.

    :param x: x matrix.
    :param keep: Percentile of values to be kept from the x matrix.
    :return: x matrix without outliers, now marked as NaN values.
    """
    for i in range(x.shape[1]):
        min_value = np.quantile(x[:, i], (1 - keep) / 2)
        max_value = np.quantile(x[:, i], (1 + keep) / 2)
        values_to_be_changed = np.logical_or(x[:, i] < min_value, x[:, i] > max_value)
        x[values_to_be_changed, i] = np.nan
    return x


def remove_useless_columns(x, stds):
    """
    Remove columns with standard deviation equals to zero.

    :param x: x matrix
    :param stds: array containing standard deviations for all feature columns.
    :return: x matrix without features having zero standard deviation.
    """
    x = np.delete(x, np.where(stds == 0), axis=1)
    stds = stds[stds != 0]
    return x, stds


def standardize(x, means, stds):
    """
    Apply standardization to a feature.

    :param x: x matrix.
    :param means: array containing means for all feature columns.
    :param stds: array containing standard deviations for all feature columns.
    :return: standardized x matrix.
    """
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - means[i]) / stds[i]
    return x
