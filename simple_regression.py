import pandas as pd
import matplotlib.pyplot as plt

# data_frame = pd.read_csv('Advertising.csv')
# x = data_frame.values[:, 2]
# y = data_frame.values[:, 4]

x = [0.01, 0.14140404, 0.181808081, 0.222212121, 0.262616162]
y = [4.88545011, 2.780649898, 3.069489813, 3.355064754, 3.637374722]


def predict(new_radio, weight, bias):
    return new_radio * weight + bias


def cost_function(x, y, weight, bias):
    n = len(x)
    cost = 0
    for i in range(n):
        cost += (y[i] - (weight * x[i] - bias)) ** 2
    return cost / n


def gradient_descent(x, y, weight, bias, learning_rate):
    n = len(x)
    weight_temp = 0
    bias_temp = 0
    for i in range(n):
        weight_temp += -2 * x[i] * (y[i] - (weight * x[i] + bias))
        bias_temp += 2 * (y[i] - (weight * x[i] + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias_temp / n) * learning_rate
    return weight, bias


def training(x, y, weight, bias, learning_rate, rounds):
    cost_his = []
    for i in range(rounds):
        weight, bias = gradient_descent(x, y, weight, bias, learning_rate)
        cost = cost_function(x, y, weight, bias)
        cost_his.append(cost)
    return weight, bias, cost_his


weight, bias, cost_his = training(x, y, 0.03, 0.0014, 0.001, 337)
print(predict(1, weight, bias))
print(predict(2, weight, bias))
print(predict(3, weight, bias))
