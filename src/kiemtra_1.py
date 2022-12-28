import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

data_frame = pd.read_csv('./data_set/kiemtra.csv')


# Cau A
def predict(input: [[]], p=1):
    x = data_frame[['A']] ** p
    y = data_frame['D']
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x.values, y.values)
    return linear_regression.coef_, linear_regression.intercept_, linear_regression.predict(input)


# Cau B
def cost_function(x, y, m, b):
    n = len(x)
    cost = 0
    for i in range(n):
        cost += (y[i] - (m * x[i] - b)) ** 2
    return cost / n


# Cau C
def on_predictive_model(input: [[]], p):
    m = np.array([0])
    x = np.array([0 for i in range(len(input))])
    for i in range(1, p + 1):
        m_temp, b_temp, x_temp = predict(input, i)
        m = m + m_temp
        x = x + x_temp
    return m, x


print(on_predictive_model([[1], [2], [3]], 2))
print(on_predictive_model([[1], [2], [3]], 3))
print(on_predictive_model([[1], [2], [3]], 4))
print(on_predictive_model([[1], [2], [3]], 5))
