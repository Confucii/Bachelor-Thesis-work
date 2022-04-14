import random
from functools import reduce
import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d
import sklearn.datasets
import argparse
import time


parser = argparse.ArgumentParser(description='Model trainer')

parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq_default', type=str)
parser.add_argument('-epochs', default=10, type=int)

args = parser.parse_args()

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MAX_LEN = 200
INPUT_SIZE = 28
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


class DatasetLFW(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.X, self.Y = sklearn.datasets.fetch_lfw_people(return_X_y=True, slice_=(slice(62, 187), slice(62, 187)), min_faces_per_person=5)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.X[idx], self.Y[idx]
        np_x = np.array(pil_x)
        np_x = np.resize(np_x, (62, 62))
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H)

        x = torch.FloatTensor(np_x)

        y = torch.LongTensor([y_idx])
        return x, y


data_loader_train, data_loader_test = torch.utils.data.random_split(DatasetLFW(), [int(len(DatasetLFW()) * 0.8), int(len(DatasetLFW()) * 0.2)])


data_loader_train = torch.utils.data.DataLoader(
    dataset=data_loader_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=data_loader_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class CCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, y, y_prim):
        self.loss = -torch.sum(
            torch.log(y_prim[range(len(y)), y] + 1e-8)
        )
        return self.loss


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=8192,
            out_features=5749
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(128, 128)) # upscale for pretrained deep model
        x = x.expand(x.size(0), 3, 128, 128) # repeat grayscale over RGB channels\
        out = self.encoder.forward(x)
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
CCE_loss = CCELoss()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, args.epochs + 1):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze()

            y_prim = model.forward(x)
            loss = CCE_loss.forward(y, y_prim)

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.array(y)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 8 == 0:
        plts = []
        c = 0
        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()