import time

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn

LEARNING_RATE = 1e-4
BATCH_SIZE = 16

X, Y = sklearn.datasets.load_boston(return_X_y=True)
# Y.shape (N, )
Y = np.expand_dims(Y, axis=1)

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))
X = X[idxes_rand]
Y = Y[idxes_rand]

idx_split = int(len(X) * 0.8)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=13, out_features=6),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=6, out_features=1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim
    #PyTorch has autograd


model = Model()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)
loss_plot_train = []
loss_plot_test = []
nrmse_plot_train = []
nrmse_plot_test = []

y_max_train = np.max(dataset_train[1])
y_min_train = np.min(dataset_train[1])
y_max_test = np.max(dataset_test[1])
y_min_test = np.min(dataset_test[1])
for epoch in range(1, 2001):

    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        nrmses = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx+BATCH_SIZE]
            y = Y[idx:idx+BATCH_SIZE]

            y_prim = model.forward(torch.FloatTensor(x)) # FloatTensor is a class like Variable
            y = torch.FloatTensor(y)
            loss = torch.mean((y - y_prim) ** 2)

            losses.append(loss.item())

            y = y.detach()
            y_prim = y_prim.detach()
            scaler = 1 / (y_max_test - y_min_test)
            if dataset == dataset_train:
                scaler = 1 / (y_max_train - y_min_train)
            nrmse = scaler * torch.sqrt(torch.mean((y - y_prim) ** 2))
            nrmses.append(nrmse)
            if dataset == dataset_train:
                loss.backward()

                optimizer.step()

        if dataset == dataset_train:
            nrmse_plot_train.append(np.mean(nrmses))
            loss_plot_train.append(np.mean(losses))
        else:
            nrmse_plot_test.append(np.mean(nrmses))
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]} nrmse_train: {nrmse_plot_train[-1]} nrmse_test: {nrmse_plot_test[-1]}')

    if epoch % 500 == 0:
        plt.subplot(2, 1, 1)
        plt.title('loss l2')
        plt.plot(loss_plot_train)
        plt.plot(loss_plot_test)
        plt.subplot(2, 1, 2)
        plt.title('nrmse')
        plt.plot(nrmse_plot_train)
        plt.plot(nrmse_plot_test)
        plt.show()