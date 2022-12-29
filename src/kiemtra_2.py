import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

data_frame = pd.read_csv('./data_set/kiemtra.csv')
x = data_frame[['A', 'B', 'C']]
y = data_frame['D']


def predict(in_put: []):
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x.values, y.values)
    result = linear_regression.predict(in_put)
    return linear_regression.coef_, linear_regression.intercept_, result


def cost_function(m, b):
    n = len(x.values)
    cost = 0
    for i in range(n):
        cost += (y[i] - (np.dot(m, x.values[i]) - b)) ** 2
    return cost / n


m, b, r = predict([[1, 1, 1], [2, 0, 4], [3, 2, 1]])

print(cost_function(m, b))
