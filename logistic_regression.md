## Logistic Regression

This chapter presents the first fully fledged example of Logistic Regression that uses commonly utilised TensorFlow stuctures.

### Data set
Data set that is used in the example comes form [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php):
Name: Breast Cancer Wisconsin (Diagnostic) Data Set (*wdbc.data* and *wdbc.names*)
Source: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

### Script

At the data preparation stage in the python script we start by reading in the data file *wdbc.data*, where first two columns names are taken from supplementary file *wdbc.names* for conviniance. Further we split the set into outcome/target and feature/predictors sets, as `ID` does not contain useful information (atleast that shuld be the case) we drop it. At this stage we have two dataframes, one for traget values of shape (569 rows x 1 columns) and one for features, which shape is (569 rows x 30 columns).

Next, we one-hot encode target lables 




### Code
 * [01_logistic_regression.py](scripts/01_logistic_regression.py)
