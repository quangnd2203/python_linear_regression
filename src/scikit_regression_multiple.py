import pandas as pd
import numpy as np
from sklearn import linear_model
import math

data_frame = pd.read_csv('./data_set/HiringProcess.csv')
x = data_frame[['Test score ', 'IQ test score', 'English score', 'Interview score', 'Years of experiences']]
y = data_frame['Salary($) per month']

linear_regression = linear_model.LinearRegression()
linear_regression.fit(x.values, y.values)

# print(linear_regression.intercept_ + np.dot(linear_regression.coef_, [5, 6, 7, 8, 1]))
