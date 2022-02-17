import random

import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d


MAX_LEN = 4000 # For debugging, reduce number of samples
BATCH_SIZE = 16

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class DatasetFassionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H)

        x = torch.FloatTensor(np_x)
        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.upper_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=in_channels),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        )

        self.lower_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=in_channels)
        )

    def forward(self, x):
        z = self.upper_layers.forward(x)
        z_prim = z + x
        z_lower = self.lower_layers.forward(z_prim)
        return z_lower

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_channels=1),
            torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            ResBlock(in_channels=2),
            torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            ResBlock(in_channels=4),
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            ResBlock(in_channels=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = torch.nn.Linear(
            in_features=16,
            out_features=10
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.layers.forward(x)
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim

model = ResNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 1000):
    plt.clf()

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)

            loss = torch.sum(-y*torch.log(y_prim + 1e-8))
            # Sum dependant on batch size => larger LR
            # Mean independant of batch size => smaller LR

            # y.to('cuda')
            # y.cuda()

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.mean((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
    if epoch % 10 == 0:
        plt.clf()
        plts = []
        c = 0
        for key, value in metrics.items():
            value = gaussian_filter1d(value, sigma=2)

            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()
