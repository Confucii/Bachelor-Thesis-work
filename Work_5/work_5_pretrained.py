import torchvision.models
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional
from tensorboardX.utils import  figure_to_image
import tensorboard_utils
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

summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class DatasetFashionMNIST(torch.utils.data.Dataset):
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

        np_x = np.expand_dims(np_x, axis=0)

        x = torch.FloatTensor(np_x)

        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=8192,
            out_features=10
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
            y = y.to(DEVICE)

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            class_num = 10
            conf_mat = np.zeros((class_num, class_num))
            for i in range(len(idx_y)):
                conf_mat[idx_y_prim[i]][idx_y[i]] += 1

            fig = plt.figure()
            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.get_cmap('Greys'))
            plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
            plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
            for x in range(class_num):
                for y in range(class_num):
                    plt.annotate(
                        str(round(100 * conf_mat[x, y] / np.sum(conf_mat[x]), 1)),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor='white'
                    )
                    plt.xlabel('True')
                    plt.ylabel('Predicted')


            tfpn_mat = np.zeros((class_num, 4))
            f1 = np.zeros(10)
            for i in range(class_num):
                for j in range(4):
                    if j == 0:
                        tfpn_mat[i, j] = conf_mat[i, i]
                    if j == 1:
                        tfpn_mat[i, j] = np.sum(
                            conf_mat[np.concatenate((np.arange(0, i), np.arange(i + 1, class_num)), dtype=np.intp), i])
                    if j == 2:
                        tfpn_mat[i, j] = np.sum(
                            conf_mat[i, np.concatenate((np.arange(0, i), np.arange(i + 1, class_num)), dtype=np.intp)])
                    if j == 3:
                        tfpn_mat[i, j] = np.sum(conf_mat[np.concatenate(
                            (np.arange(0, i), np.arange(i + 1, class_num))), np.expand_dims(
                            np.concatenate((np.arange(0, i), np.arange(i + 1, class_num))), axis=1)])
                f1[i] = 2 * tfpn_mat[i, 0] / (2 * tfpn_mat[i, 0] + tfpn_mat[i, 1] + tfpn_mat[i, 2] + 1e-8)
            f1_fin = np.mean(f1)

            if data_loader == data_loader_train:
                summary_writer.add_figure(
                    tag='Confusion Matrix Train',
                    figure=fig,
                    global_step=epoch
                )
                summary_writer.add_scalar(
                    tag='F1 Train',
                    scalar_value=f1_fin,
                    global_step=epoch
                )

            else:
                summary_writer.add_figure(
                    tag='Confusion Matrix Test',
                    figure=fig,
                    global_step=epoch
                )
                summary_writer.add_scalar(
                    tag='F1 Test',
                    scalar_value=f1_fin,
                    global_step=epoch
                )

            summary_writer.add_hparams(
                hparam_dict={'Learning rate':LEARNING_RATE, 'Batch size':BATCH_SIZE},
                metric_dict={'Best accuracy': acc},
                name=args.run_name,
                global_step=epoch
            )

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                summary_writer.add_scalar(
                    tag=key,
                    scalar_value=value,
                    global_step=epoch
                )
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 10 == 0:
        plts = []
        c = 0
        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()
    summary_writer.flush()
summary_writer.close()