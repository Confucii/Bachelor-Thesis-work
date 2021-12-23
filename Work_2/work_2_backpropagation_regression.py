import numpy as np

import matplotlib

import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

W1 = 1
b1 = 1

def linear(W, b, x):
    return W * x + b


def dW_linear(W, b, x):
    return x


def db_linear(W, b, x):
    return 1


def dx_linear(W, b, x):
    return W


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def dx_tanh(x):
    return 4 / ((np.exp(x) + np.exp(-x)) ** 2)


def LeakyReLU(z):
    y = np.array(z)
    a = 5
    y[y <= 0] *= a
    return y


def dz_LeakyReLU(z):
    y = np.array(z)
    a = 5
    y[y > 0] = 1
    y[y <= 0] = a
    return y


def model(W1, b1, W, b, x):
    return LeakyReLU(linear(W1, b1, tanh(linear(W, b, x))))


def dW_model(W1, b1, W, b, x):
    return dz_LeakyReLU(linear(W1, b1, tanh(linear(W, b, x)))) * dx_linear(W1, b1, tanh(linear(W, b, x))) * dx_tanh(linear(W, b, x)) * dW_linear(W, b, x)


def db_model(W1, b1, W, b, x):
    return dz_LeakyReLU(linear(W1, b1, tanh(linear(W, b, x)))) * dx_linear(W1, b1, tanh(linear(W, b, x))) * dx_tanh(linear(W, b, x)) * db_linear(W, b, x)


def dW1_model(W1, b1, W, b, x):
    return dz_LeakyReLU(linear(W1, b1, tanh(linear(W, b, x)))) * dW_linear(W1, b1, tanh(linear(W, b, x)))


def db1_model(W1, b1, W, b, x):
    return dz_LeakyReLU(linear(W1, b1, tanh(linear(W, b, x)))) * db_linear(W1, b1, tanh(linear(W, b, x)))


def loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)


def dW_loss(y, y_prim, W1, b1, W, b, x):
    return np.mean(-2 * (y - y_prim) * dW_model(W1, b1, W, b, x))


def db_loss(y, y_prim, W1, b1, W, b, x):
    return np.mean(-2 * (y - y_prim) * db_model(W1, b1, W, b, x))


def dW1_loss(y, y_prim, W1, b1, W, b, x):
    return np.mean(-2 * (y - y_prim) * dW1_model(W1, b1, W, b, x))


def db1_loss(y, y_prim, W1, b1, W, b, x):
    return np.mean(-2 * (y - y_prim) * db1_model(W1, b1, W, b, x))


learning_rate = 1e-2
for epoch in range(7000):
    Y_prim = model(W1, b1, W, b, X)
    loss1 = loss(Y, Y_prim)

    dW_loss1 = dW_loss(Y, Y_prim, W1, b1, W, b, X)
    db_loss1 = db_loss(Y, Y_prim, W1, b1, W, b, X)
    dW_loss2 = dW1_loss(Y, Y_prim, W1, b1, W, b, X)
    db_loss2 = db1_loss(Y, Y_prim, W1, b1, W, b, X)

    W -= dW_loss1 * learning_rate
    b -= db_loss1 * learning_rate
    W1 -= dW_loss2 * learning_rate
    b1 -= db_loss2 * learning_rate

    print(f'Y_prim: {Y_prim},    loss: {loss1}')

print(f'Floor 4 = {model(W1, b1, W, b, np.array([4]))}')
print(f'Floor 6 = {model(W1, b1, W, b, np.array([6]))}')