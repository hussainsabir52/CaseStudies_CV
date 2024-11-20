from helper import time_series_cv
import pandas as pd

data =  pd.read_csv("Datasets/Timeseries Dataset/Microsoft_Stock.csv")

train_size, test_size = 60, 20
partitions = time_series_cv.rolling_window_partition_df(data, train_size, test_size)

first_train, first_test = partitions[0]
first_train.head(), first_test.head()
print(first_train)