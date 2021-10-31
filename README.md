# ML project 1 - Novae
This project is the Project 1 of **EPFL CS-433 Machine Learning**. The project is the same as the **Higgs Boson Machine Learning Challenge** as that on [Kaggle](https://www.kaggle.com/c/higgs-boson).

The dataset and the detailed description can also be found at [the GitHub respository of the course](https://github.com/epfml/ML_course/tree/master/projects/project1).

## Instruction
* The project runs under `Python 3.?` and requires `NumPy`.
* Please ensure that `test.csv` is placed inside the `data` folder. If not, please download it from the course website or Kaggle mentioned above.

* Go to the `script/` folder and execute `run.py`.

---
## Modules
### `implementations.py`
Contains the implementations of different learning algorithms. Including
* Least squares linear regression
    * Direct computation from linear equations.
    * Gradient descent.
    * Stochastic gradient descent.
    * Regularized linear regression from direct computation.
* Logistic regression
    * Gradient descent
    * Gradient descent with regularization.

### `feature_engineering.py`
Contains the functions used to process data and generate our features.
### `run.py`
Generates the submission `.csv` file based on the `test.csv` stored in the folder `data`