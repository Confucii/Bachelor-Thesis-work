import time

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-3
BATCH_SIZE = 5

X, Y = sklearn.datasets.load_wine(return_X_y=True)

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))
X = X[idxes_rand]
Y = Y[idxes_rand]

Y_idxes = Y
Y = np.zeros((len(Y), 3))
Y[np.arange(len(Y)), Y_idxes] = 1.0

idx_split = int(len(X) * 0.9)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))


class Variable:
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.normal(size=(out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output # this output will be input for next function in model

    def backward(self):
        # W*x + b / d b = 0 + b^{1-1} = 1
        # d_b = 1 * chain_rule_of_prev_d_func
        self.b.grad = 1 * self.output.grad

        # d_W = x * chain_rule_of_prev_d_func
        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        # d_x = W * chain_rule_of_prev_d_func
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable((x.value >= 0) * x.value)
        # [1, -1, -1] >= 0
        # [True, False, False] * 1.0 => [1, 0, 0]
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad

class LayerSoftmax():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        np_x = np.copy(x.value)
        np_x -= np.max(np_x, axis=1, keepdims=True)
        self.output = Variable(
            (np.exp(np_x + 1e-8) / np.sum(np.exp(np_x), axis=1, keepdims=True))
        )
        return self.output

    def backward(self):
        J = np.zeros((BATCH_SIZE, 3, 3))
        a = self.output.value
        for i in range(3):
            for j in range(3):
                if i == j:
                    J[:, i, j] = a[:, i] * (1 - a[:, j])
                else:
                    J[:, i, j] = -a[:, i] * a[:, j]

        self.x.grad = np.matmul(J, np.expand_dims(self.output.grad, axis=2)).squeeze()


class LossCrossEntropy():
    def __init__(self):
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim
        #self.y_prim.value[self.y_prim.value == 0] = 1e-3 # I have tried different values here.
        return np.mean(-y.value * np.log(y_prim.value))

    def backward(self):
        self.y_prim.grad = -self.y.value / self.y_prim.value


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=13, out_features=8), # W_1*x + b < dW , db
            LayerReLU(),
            LayerLinear(in_features=8, out_features=4), # W_2*x + b
            LayerReLU(),
            LayerLinear(in_features=4, out_features=3), # W_3*x + b
            LayerSoftmax()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W (8, 4)            dW (16, 8, 4)  => (Batch, InFeatures, OutFeatures)
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate  # W = W - dW * alpha

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossCrossEntropy()


loss_train = []
loss_test = []
acc_train = []
acc_test = []

for epoch in range(1, 10000):

    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        accuracies = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx+BATCH_SIZE]
            y = Y[idx:idx+BATCH_SIZE]

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)

            accuracy = 0
            for i in range(BATCH_SIZE):
                y_max = np.argmax(y[i])
                y_prim_max = np.argmax(y_prim.value[i])
                if y_max == y_prim_max:
                    accuracy += 1
            accuracy /= BATCH_SIZE

            losses.append(loss)
            accuracies.append(accuracy)

            if dataset == dataset_train:
                loss_fn.backward()
                model.backward()
                optimizer.step()

        if dataset == dataset_train:
            acc_train.append(np.mean(accuracies))
            loss_train.append(np.mean(losses))
        else:
            acc_test.append(np.mean(accuracies))
            loss_test.append(np.mean(losses))

    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_train[-1]} '
        f'loss_test: {loss_test[-1]} '
        f'acc_train: {acc_train[-1]} '
        f'acc_test: {acc_test[-1]} '
    )

    if epoch % 1000 == 0:
        plt.subplot(2, 1, 1)
        plt.title('loss')
        plt.plot(loss_train)
        plt.plot(loss_test)

        plt.subplot(2, 1, 2)
        plt.title('acc')
        plt.plot(acc_train)
        plt.plot(acc_test)
        plt.show()