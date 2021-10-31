# ML project 1 - Novae
This project is the Project 1 of **EPFL CS-433 Machine Learning**. The project is the same as the **Higgs Boson Machine Learning Challenge** as that on [Kaggle](https://www.kaggle.com/c/higgs-boson).

The dataset and the detailed description can also be found at [the GitHub respository of the course](https://github.com/epfml/ML_course/tree/master/projects/project1).

## Instruction
* Please ensure that `test.csv` is placed inside the `data` folder. If not, please download it from the course website or Kaggle mentioned above.

* Go to the `script/` folder and execute `run.py`.

---

## Data Preparation
Observe the value of the raw feature *PRI_JET_NUM*, we can first divide the dataset into 3 different subgroups. We can eliminate most of the missing entries by this technique.

The only remaining entry which still possibly has missing values is the first column *DER_mass_MMC*. We simply replaced the median of the respective feature. Although we tried another attempt, we end up decided to adopt this simple method.

We generated our features in the meaning of energy square by computing the polynomial expansion of the mass, energy, and momentum terms. (more detail can be found in the report file). So that our features can be practically interpreted.

After the this feature generation for 3 subgroups, we identified the outliers by selecting feature values lower than 0.025-th or higher than 0.975-th quantile. We replaced these outlier values with the median.

We then standardized each feature after all the above processing is done. 

## Model Selection
After tried both least squares linear regression and logistic regression, we found out the linear regression performs better on this task even though it's a classification problem.

## Validation
We use 4-fold cross validation to test our model. We obtained 0.82 of accuracy on the test set and 0.76 F1 score using the following hyper parameters.

| jet_num | $\lambda$ | Degree |
| :---: | :---: | :---: |
| 0     | 1e-2  | 7     |
| 1     | 1e-5  | 2     |
| 2,3   | 1e-3  | 6     |