from data_processing import prepare_train_data, prepare_test_data
from models import compute_weights, compute_predictions
from proj1_helpers import create_csv_submission, load_csv_data

models = {
    "jet0" : {
        "model" : "least squares",
        "degree" : 4,
        "lambda" : 1e-10,
        "mixed" : True,
        "accuracy" : 0.8470
    },
    "jet1" : {
        "model" : "least squares",
        "degree" : 6,
        "lambda" : 1e-3,
        "mixed" :  True,
        "accuracy" : 0.8069
    },
    "jet2" : {
        "model" : "least squares",
        "degree" : 4,
        "lambda" : 1e-4,
        "mixed" : True,
        "accuracy" : 0.8342
    }
}



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