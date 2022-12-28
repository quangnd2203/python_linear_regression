import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


data_frame = pd.read_csv('./data_set/ChirpsPerMinute.csv')
x = data_frame[['Cricket chirps per Minute ']]
y = data_frame['Temperature']

linear_regression = linear_model.LinearRegression()
linear_regression.fit(x, y)

m = linear_regression.coef_
b = linear_regression.intercept_

plt.scatter(data_frame['Cricket chirps per Minute '], data_frame['Temperature'])
plt.plot(data_frame['Cricket chirps per Minute '], m * data_frame['Cricket chirps per Minute '] + b)
plt.plot(150, 150 * m + b, marker="o", markersize=20, markeredgecolor="red")
plt.show()

# print(linear_regression.coef_)
# print(linear_regression.intercept_)
#
# print(linear_regression.predict([[300]]))
