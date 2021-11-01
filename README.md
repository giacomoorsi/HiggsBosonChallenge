# ML project 1 - Novae
This project is the Project 1 of **EPFL CS-433 Machine Learning**. The project is the same as the **Higgs Boson Machine Learning Challenge** posted on [Kaggle](https://www.kaggle.com/c/higgs-boson). The dataset and the detailed description can also be found in [the GitHub repository of the course](https://github.com/epfml/ML_course/tree/master/projects/project1).

`Team name`: Novae

`Team member`: G. Orsi, V. Rossi, C. Tsai
## About the Project

The task of this project is to train a model based on the provided `train.csv` to have the best prediction on the data given in `test.csv` or any other general case.

We built our model for the problem using regularized linear regression after applying some data cleaning and features engineering techniques. In the end, we obtained an accuracy of 0.82 and an F1 score of 0.76 on the `test.csv` dataset.
## Instruction
* The project runs under `Python 3.8` and requires `NumPy=1.19`.
* Please make sure to place `test.csv` inside the `data` folder. If not, please download it from the course website or Kaggle mentioned above.

* Go to the `script/` folder and execute `run.py`. A submission file will be generated as `out.csv`.

---
## Modules
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

There are also some helper functions in this file to facilitate the above functions.

### `data_processing.py`
Calls the following files to process the data.
* `data_cleaning.py`:
Contains functions used to
    1. Categorize data into subgroups.
    2. Replace missing values and outliers with the median.
    3. Standardize the features.
* `feature_engineering.py`:
Contains functions used to generate our interpretable features.

### `run.py`
Generates the submission `.csv` file based on the data of `test.csv` stored in the folder `data/`. Our optimized model is also defined in this file.


### **Some helper Functions**
* `models.py`: 
Create the models for predicting the labels for new data points without true labels.
* `expansions.py`: 
Contains a function to apply polynomial expansion to our features to add extra degrees of freedom for our models.
* `proj1_helpers.py`: 
Contains functions which loads the `.csv` files as training or testing data, and create the `.csv` file for submission.
* `cross_validation.py`:
Contains a function to build the index for k-fold cross_validation.
* `disk_helper.py`:
Save/load the NumPy array to disk for further usage. Useful for saving hyper-parameters when trying a long training process.