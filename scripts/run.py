from data_processing import prepare_train_data, prepare_test_data
from models import compute_weights, compute_predictions
from proj1_helpers import create_csv_submission, load_csv_data

models =  {
    "jet0" : { # the subset of the dataset
        "model" : "least squares", # the model to be used ["least squares" or "logistic regression"]
        "degree" : 6, # the degree of polynomial expansion to be used
        "lambda" : 1e-7, # the lambda to be used in the ridge regression
        "mixed" : True, # if true, pair products of the features will be added to the dataset (but will not be elevated in polynomial expansion)
        "accuracy" : 0.8476 # accuracy on a 4-fold cross validation on train dataset, just for reference
    },
    
    "jet1" : {
        "model" : "least squares",
        "degree" : 7,
        "lambda" : 1e-6,
        "mixed" :  True,
        "accuracy" : 0.8193  
    },
    "jet2" : {
        "model" : "least squares",
        "degree" : 7,
        "lambda" : 1e-5,
        "mixed" : True,
        "accuracy" : 0.8378 
    }
} # Accuracy on test dataset on AICrowd: 0.836, F1: 0.751 



def main() : 
    DATA_TRAIN_PATH = '../data/train.csv'
    DATA_TEST_PATH = '../data/test.csv'
    JET_COLUMN_INDEX = 22
    OUTPUT_PATH = "out.csv"

    y, tX, _ = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
    y_jets, x_jets, means, stds, replacing_values = prepare_train_data(y, tX)

    w_jets = compute_weights(x_jets, y_jets, models)

    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH, sub_sample=False)
    x_test_jets = prepare_test_data(tX_test, means, stds, replacing_values)

    y_pred = compute_predictions(x_test_jets, w_jets, models, tX_test[:, JET_COLUMN_INDEX])

    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


if __name__ == "__main__" : main()