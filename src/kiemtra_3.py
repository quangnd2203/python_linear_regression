import matplotlib.pyplot as plt


def f(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2


def dx(x):
    return 2 * x - 4


def dy(y):
    return 2 * y - 6


def gradient_descent(x0, y0, leaning_rate, time):
    x_history = []
    x = x0
    y_history = []
    y = y0
    for i in range(time):
        x -= dx(x0) * leaning_rate
        y -= dy(y0) * leaning_rate
        x_history.append(x)
        y_history.append(y)
    return x, y, x_history, y_history


def show(x_history, y_history):
    r = []
    for i in range(len(x_history)):
        r.append(f(x_history[i], y_history[i]))
    return r


x, y, x_history, y_history = gradient_descent(0, 0, 0.5, 10)

r = show(x_history, y_history)

ax = plt.axes(projection='3d')
ax.scatter(x_history, y_history, r)

plt.xlabel('x')
plt.ylabel('y')

plt.show()
