# ML project 1 - Novae
This project is the Project 1 of **EPFL CS-433 Machine Learning**. The project is the same as the **Higgs Boson Machine Learning Challenge** as that on [Kaggle](https://www.kaggle.com/c/higgs-boson).

The dataset and the detailed description can also be found at [the GitHub respository of the course](https://github.com/epfml/ML_course/tree/master/projects/project1).

## Instruction
* The project runs under `Python 3.8?` and requires `NumPy=1.19`.
* Please ensure that `test.csv` is placed inside the `data` folder. If not, please download it from the course website or Kaggle mentioned above.

* Go to the `script/` folder and execute `run.py`.

---
## Modules
### `proj1_helpers.py`
Contains functions which loads the `.csv` files as training or testing data, and create the `.csv` file for submission.
### `implementations.py`
Contains the implementations of different learning algorithms. Including
* Least squares linear regression
    * `least_squares`: Direct computation from linear equations.
    * `least_squares_GD`: Gradient descent.
    * `least_squares_SGD`: Stochastic gradient descent.
    * `ridge_regression`: Regularized linear regression from direct computation.
* Logistic regression
    * `logistic_regression`: Gradient descent
    * `reg_logistic_regression`: Gradient descent with regularization.

There are also some small functions in this file to facilitate the functions above.

### `data_processing.py`
Calls the following files to process the data.
* `data_cleaning.py`
Contains functions used to
    1. Categorize data into subgroups.
    2. Replace missing values and outliers with median.
    3. Standardize the features.
* `feature_engineering.py`
Contains functions used to generate our interpretable features.

### `models.py`
Create the models for predicting the labels for new data points without true labels.

### `expansions.py`
Contains a function to apply polynomial expansion to our features to add extra degree of freedom for our models.

### `helper.py`
The batch iteration function for the function `least_squares_SGD` in `implementations.py`. Also contains other functions for ploting figures. Note that we need `matplotlib`if you want to use these ploting functions.

### `run.py`
Generates the submission `.csv` file based on the `test.csv` stored in the folder `data`