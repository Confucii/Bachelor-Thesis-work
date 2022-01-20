import time

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn

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


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=13, out_features=8), # W_1*x + b < dW , db
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=8), # W_2*x + b
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=3), # W_3*x + b
            torch.nn.Softmax()
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)



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

            y_prim = model.forward(torch.FloatTensor(x))
            y = torch.FloatTensor(y)
            loss = torch.mean(-y * torch.log(y_prim))

            losses.append(loss.item())

            y = y.detach()
            y_prim = y_prim.detach()

            y = y.tolist()
            y_prim = y_prim.tolist()
            accuracy = 0
            for i in range(BATCH_SIZE):
                y_max = np.argmax(y[i])
                y_prim_max = np.argmax(y_prim[i])
                if y_max == y_prim_max:
                    accuracy += 1
            accuracy /= BATCH_SIZE

            accuracies.append(accuracy)

            if dataset == dataset_train:
                loss.backward()
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