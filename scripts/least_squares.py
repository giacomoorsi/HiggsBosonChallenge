import numpy as np
import math
from cross_validation import build_k_indices
from proj1_helpers import predict_labels


def ridge_regression(y, tx, lambda_) :
    n, d = tx.shape
    w = np.zeros(d)
    w = np.linalg.solve(tx.T @ tx + n * lambda_ * np.eye(d), tx.T @ y)
    return w, 1 / (2 * n) * np.sum((y - tx @ w) ** 2) + lambda_ / 2 * w.T.dot(w)

def cross_validate_least_squares(y, tx, lambda_, k_fold) : 
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    classifications = []

    for k in range(k_fold) : 
        test_indices = np.zeros(len(y)).astype(bool)
        test_indices[k_indices[k]] = True
        train_indices = (~test_indices).tolist()
        test_indices = test_indices.tolist()
            
        tx_train = tx[train_indices, :]
        y_train = y[train_indices]
        
        tx_test = tx[test_indices, :]
        y_test = y[test_indices]
        
        w, err = ridge_regression(y_train, tx_train, lambda_)
        classified = sum(predict_labels(w, tx_test)==y_test)/len(y_test)

        #print("Fold: {k}, lambda_: {l}, acc={a}".format(k=k, l=lambda_, a=classified))
        classifications.append(classified)

    return np.mean(classifications)