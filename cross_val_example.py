import numpy as np
from helper.classifications_cv import get_k_folds_splits_and_train
from sklearn.linear_model import LinearRegression


model = LinearRegression()

result = get_k_folds_splits_and_train("Datasets/Regression Datasets/synthetic_data.csv",{"n_splits":2},model,"regression")

print(result)