from data_processing import prepare_train_data, prepare_test_data
from models import compute_weights, compute_predictions
from proj1_helpers import create_csv_submission, load_csv_data

models = {
 "jet0" : {
        "model" : "least squares",
        "degree" : 7,
        "lambda" : 1e-5,
        "mixed" : False,
    },
    "jet1" : {
        "model" : "least squares",
        "degree" : 7,
        "lambda" : 1e-4,
        "mixed" : False,
    },
    "jet2" : {
        "model" : "least squares",
        "degree" : 7,
        "lambda" : 1e-4,
        "mixed" : False,
    }
}



def main() : 
    DATA_TRAIN_PATH = '../data/train.csv'
    DATA_TEST_PATH = '../data/test.csv'
    JET_COLUMN_INDEX = 22
    OUTPUT_PATH = "out.csv"

    y, tX, _ = load_csv_data(DATA_TRAIN_PATH)
    x_jets, y_jets, replacing_values, means, stds = prepare_train_data(tX, y)

    w = compute_weights(x_jets, y_jets, models)

    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    tX_test_jets = prepare_test_data(replacing_values, means, stds)

    y_pred = compute_predictions(tX_test[:, JET_COLUMN_INDEX], tX_test, w)

    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)