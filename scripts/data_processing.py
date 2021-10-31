import numpy as np
import data_cleaning as dc
import feature_engineering as fe


def prepare_train_data(y, x):
    print("Preparing training data...")
    y_jets, x_jets = dc.split_train_data(y, x)

    # Arrays containing statistical information about the
    # training dataset. Saved to use on the testing set
    medians, stds, means = [], [], []
    for i, x_jet in enumerate(x_jets):
        print("Removing values and dealing with outliers in tX with jet number {}...".format(i))
        x_jet = dc.replace_missing_with_nan(x_jet)
        x_jet = dc.drop_x_features(x_jet, i)
        x_jet = dc.replace_outliers_with_nan(x_jet)
        # Compute medians and use them in place of NaNs
        medians.append(np.nanmedian(x_jet, axis=0))
        x_jet = dc.replace_nan_with_medians(x_jet, medians[i])
        # Apply feature engineering methods
        x_jet = fe.feature_expand(x_jet, i)
        stds.append(np.std(x_jet, axis=0))
        x_jet, stds[i] = dc.remove_useless_columns(x_jet, stds[i])
        means.append(np.mean(x_jet, axis=0))
        print("Standardizing...")
        x_jet = dc.standardize(x_jet, means[i], stds[i])
        x_jets[i] = x_jet
    return y_jets, x_jets, means, stds, medians


def prepare_test_data(x, means, stds, medians):
    print("Preparing testing data...")
    x_jets = dc.split_test_data(x)

    for i, x_jet in enumerate(x_jets):
        print("Replacing missing values in tX with jet number {}...".format(i))
        x_jet = dc.replace_missing_with_nan(x_jet)
        x_jet = dc.drop_x_features(x_jet, i)
        x_jet = dc.replace_nan_with_medians(x_jet, medians[i])
        # Apply feature engineering methods
        x_jet = fe.feature_expand(x_jet, i)
        print("Standardizing...")
        x_jet = dc.standardize(x_jet, means[i], stds[i])
        x_jets[i] = x_jet
    return x_jets