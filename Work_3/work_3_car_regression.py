import os
import pickle
import time
import matplotlib
import sys
import numpy as np
import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-2
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


def normalize(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    return (x - (x_max + x_min) * 0.5) / ((x_max - x_min) * 0.5)

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        self.X = np.array(self.X).astype(np.float)
        self.X = normalize(self.X)

        self.Y = np.array(self.Y).astype(np.float)
        self.Y = normalize(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), self.Y[idx]

class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if len(self) < self.idx_batch:
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        batch = self.dataset[idx_start:idx_end]
        X, Y = batch
        Y = np.expand_dims(Y, axis=-1)
        self.idx_batch += 1
        return X, Y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value= ( np.random.random(size=(in_features, out_features)) - 0.5 ) * 2.0,
            grad = np.zeros((BATCH_SIZE, in_features, out_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,)),
            grad=np.zeros((BATCH_SIZE, out_features)),
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        x_3d = np.expand_dims(x.value, axis=-1)
        Wx_3d = self.W.value.T @ x_3d
        Wx_2d = np.squeeze(Wx_3d, axis=-1)
        self.output = Variable(
            Wx_2d + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)



class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            1.0 / (1.0 + np.exp(-x.value))
        )
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1.0 - self.output.value) * self.output.grad


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            value=np.array(x.value)
        )
        self.output.value[self.output.value < 0] = 0
        return self.output

    def backward(self):
        self.x.grad += (self.x.value >= 0) * self.output.grad


class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean( np.power( (self.y.value - self.y_prim.value), 2 ) )
        return loss

    def backward(self):
        self.y_prim.grad += -2 * (self.y.value - self.y_prim.value)


class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad += -(self.y.value - self.y_prim.value) / np.abs((self.y.value - self.y_prim.value))


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=7, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=1)
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
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


class NRMSE:
    def __init__(self):
        self.y = None
        self.y_prim = None
        self.MSE = LossMSE()

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        RMSE = np.sqrt( self.MSE.forward(self.y, self.y_prim) )
        std = np.sqrt( np.sum(np.power((self.y.value - np.mean(self.y_prim.value)), 2)) / (self.y.value.size - 1) )
        loss = RMSE / std
        return loss

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMSE()
loss_nrmse = NRMSE()


loss_plot_train = []
loss_plot_test = []

nrmse_plot_train = []
nrmse_plot_test = []
for epoch in range(1, 1001):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        losses_2 = []
        for x, y in dataloader:

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)

            loss_2 = loss_nrmse.forward(Variable(value=y), y_prim)

            losses.append(loss)
            losses_2.append(loss_2)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            nrmse_plot_train.append(np.mean(losses_2))
        else:
            loss_plot_test.append(np.mean(losses))
            nrmse_plot_test.append(np.mean(losses_2))

    print(f'epoch: {epoch}\nloss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}\nnrmse_train: {nrmse_plot_train[-1]} nrmse_test: {nrmse_plot_test[-1]}\n')

    if epoch % 50 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(nrmse_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(nrmse_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()