import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

data_frame = pd.read_csv('./data_set/kiemtra.csv')


# Cau A
def predict(in_put: [[]], p=1, is_use_b=True, ):
    x = data_frame[['A']] ** p
    y = data_frame['D']
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x.values, y.values)
    if is_use_b:
        result = linear_regression.predict(in_put)
    else:
        result = linear_regression.coef_ * np.array(in_put).reshape(-1)
    return linear_regression.coef_, linear_regression.intercept_, result


# Cau B
def cost_function(x, y, m, b):
    n = len(x)
    cost = 0
    for i in range(n):
        cost += (y[i] - (m * x[i] - b)) ** 2
    return cost / n


# Cau C
def on_predictive_model(in_put: [[]], p):
    m = np.array([0])
    x = np.array([0 for i in range(len(in_put))])
    for i in range(1, p + 1):
        m_temp, b_temp, x_temp = predict(in_put, i, False)
        m = m + m_temp
        x = x + x_temp
    return m, x


print(predict([[1], [2], [3]]))
