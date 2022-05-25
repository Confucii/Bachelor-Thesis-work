import matplotlib.pyplot as plt
import torchvision.models
import torch
import numpy as np
import torchvision
import torch.utils.data
import torch.nn.functional
import tensorboard_utils as tu
import argparse
import time
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Model trainer')

parser.add_argument('-run_name', default=f'run', type=str)
parser.add_argument('-sequence_name', default=f'seq', type=str)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=5, type=int)
parser.add_argument('-len', default=0, type=int)
parser.add_argument('-sm', default=0.3, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-dataset', default='FASHION', type=str)
parser.add_argument('-loss', default='JS', type=str)
parser.add_argument('-pdf', default='SOFT', type=str)
parser.add_argument('-gamma', default=1, type=float)
parser.add_argument('--local_rank', default=0, type=int)

args = parser.parse_args()

MAX_LEN = args.len
if not torch.cuda.is_available():
    args.device = 'cpu'
    MAX_LEN = 500


if args.dataset == 'FASHION':
    class DatasetFashionMNIST(torch.utils.data.Dataset):
        def __init__(self, is_train):
            super().__init__()
            if is_train:
                self.data = torchvision.datasets.FashionMNIST(
                    root='./data',
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(0.5),
                        torchvision.transforms.RandomVerticalFlip(0.5),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.2))
                        ]), p=0.9),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                        ]), p=0.4)
                    ]),
                    train=is_train,
                    download=True
                )
            else:
                self.data = torchvision.datasets.FashionMNIST(
                    root='./data',
                    train=is_train,
                    download=True
                )

            self.class_count = 10

        def __len__(self):
            if MAX_LEN:
                return MAX_LEN
            return len(self.data)

        def __getitem__(self, idx):
            # list tuple np.array torch.FloatTensor
            pil_x, y_idx = self.data[idx]
            np_x = np.array(pil_x, dtype=np.float32) / 255.0

            np_x = np.expand_dims(np_x, axis=0)
            x = torch.FloatTensor(np_x)
            y = torch.LongTensor([y_idx])
            return x, y

    set_train = DatasetFashionMNIST(is_train=True)
    set_test = DatasetFashionMNIST(is_train=False)

elif args.dataset == 'CIFAR':
    class DatasetCIFAR(torch.utils.data.Dataset):
        def __init__(self, is_train):
            super().__init__()
            if is_train:
                self.data = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=is_train,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(0.5),
                        torchvision.transforms.RandomVerticalFlip(0.5),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.2))
                        ]), p=0.9),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                        ]), p=0.4)
                    ]),
                    download=True
                )
            else:
                self.data = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=is_train,
                    download=True
                )

            self.class_count = 10

        def __len__(self):
            if MAX_LEN:
                return MAX_LEN
            return len(self.data)

        def __getitem__(self, idx):
            # list tuple np.array torch.FloatTensor
            pil_x, y_idx = self.data[idx]
            np_x = np.array(pil_x, dtype=np.float32) / 255.0

            np_x = np.transpose(np_x, (2, 0, 1))

            x = torch.FloatTensor(np_x)
            y = torch.LongTensor([y_idx])
            return x, y

    set_train = DatasetCIFAR(is_train=True)
    set_test = DatasetCIFAR(is_train=False)

elif args.dataset == "KMNIST":
    class DatasetKMNIST(torch.utils.data.Dataset):
        def __init__(self, is_train):
            super().__init__()
            if is_train:
                self.data = torchvision.datasets.KMNIST(
                    root='./data',
                    train=is_train,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(0.5),
                        torchvision.transforms.RandomVerticalFlip(0.5),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.2))
                        ]), p=0.9),
                        torchvision.transforms.RandomApply(torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                        ]), p=0.4)
                    ]),
                    download=True
                )
            else:
                self.data = torchvision.datasets.KMNIST(
                    root='./data',
                    train=is_train,
                    download=True
                )

            self.class_count = 10

        def __len__(self):
            if MAX_LEN:
                return MAX_LEN
            return len(self.data)

        def __getitem__(self, idx):
            # list tuple np.array torch.FloatTensor
            pil_x, y_idx = self.data[idx]
            np_x = np.array(pil_x, dtype=np.float32) / 255.0

            np_x = np.expand_dims(np_x, axis=0)
            x = torch.FloatTensor(np_x)
            y = torch.LongTensor([y_idx])
            return x, y

    set_train = DatasetKMNIST(is_train=True)
    set_test = DatasetKMNIST(is_train=False)


data_loader_train = torch.utils.data.DataLoader(
    dataset=set_train,
    batch_size=args.batch_size,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=set_test,
    batch_size=args.batch_size,
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


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.loss = 0

    def forward(self, y, y_prim):
        self.loss = -torch.sum(
            (((1 - y_prim[range(len(y)), y] + 1e-8) ** self.gamma) * torch.log(y_prim[range(len(y)), y] + 1e-8)))
        return self.loss


class KLDivergence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, y, y_prim):
        self.loss = torch.sum(
            torch.log(1 / (y_prim[range(len(y)), y] + 1e-8))
        )
        return self.loss


class JSDivergence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, y, y_prim):
        M = torch.zeros_like(y_prim)
        M[range(len(y)), y] = 1
        y_ch = 0.5 * (M + y_prim)
        self.loss = torch.sqrt(0.5 * (torch.sum(1 / (y_ch[range(len(y)), y] + 1e-8)) + torch.sum(y_prim * torch.log((y_prim + 1e-8) / (y_ch + 1e-8)))))
        return self.loss


class SMSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, dist):
        tens = torch.rand_like(logits)
        max, ind = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max
        for sub in range(logits.size(dim=0)):
            for i in range(logits.size(dim=1)):
                tens[sub, i] = torch.exp(logits[sub, i] - dist + 1e-8) / (torch.exp(logits[sub, i] - dist + 1e-8) + torch.sum(torch.exp(logits[sub, :] + 1e-8)) - torch.exp(logits[sub, i] + 1e-8) + 1e-8)
        return tens


class TaylorSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        tens = torch.rand_like(logits)
        max, ind = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max
        for sub in range(logits.size(dim=0)):
            for i in range(logits.size(dim=1)):
                tens[sub, i] = (1 + logits[sub, i] + torch.pow(logits[sub, i] + 1e-8, 2) / 2) / (torch.sum(1 + logits[sub, :] + torch.pow(logits[sub, :] + 1e-8, 2) / 2) + 1e-8)
        return tens


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.fc = torch.nn.Linear(
            in_features=1000,
            out_features=set_train.class_count
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear')
        if args.dataset != 'CIFAR':
            x = x.expand(batch_size, 3, x.size(2), x.size(3)) # repeat grayscale over RGB channels\
        out = self.encoder.forward(x)
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        if args.pdf == 'SMSOFT':
            y_prim = SMSoftmax.forward(self, logits, args.sm) #soft-margin softmax
        elif args.pdf == 'TSOFT':
            y_prim = TaylorSoftmax.forward(self, logits) #taylor softmax
        else:
            y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if args.loss == 'CCE':
    f_loss = CCELoss()
elif args.loss == 'FOCAL':
    f_loss = FocalLoss(args.gamma)
elif args.loss == "KL":
    f_loss = KLDivergence()
elif args.loss == "JS":
    f_loss = JSDivergence()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

summary_writer = tu.CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}_{int(time.time())}'
)

best_loss = np.empty(args.epochs)
best_acc = np.empty(args.epochs)

for epoch in range(1, args.epochs + 1):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}
        conf_matrix = np.zeros((set_train.class_count, set_train.class_count))

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in tqdm(data_loader, desc=stage):
            x = x.to(args.device)
            y = y.to(args.device).squeeze()

            y_prim = model.forward(x)
            loss = f_loss.forward(y, y_prim)

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((np_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            for idx in range(np_y.size):
                conf_matrix[idx_y_prim[idx], np_y[idx]] += 1

        fig = plt.figure()
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        for x in range(set_train.class_count):
            for y in range(set_train.class_count):
                perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix[x]), 1)
                plt.annotate(
                    str(perc),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor=(1., 1., 1., 0.),
                    color='black' if perc < 50 else 'white',
                    fontsize=7
                )
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.tight_layout(pad=0)

        summary_writer.add_figure(
            figure=fig,
            tag=f'{stage}_conf_matrix',
            global_step=epoch
        )

        metrics_mean = {}
        for key in metrics_epoch:
            if stage in key:
                mean_value = np.mean(metrics_epoch[key])
                if key == 'test_acc':
                    best_acc[epoch - 1] = mean_value
                elif key == 'test_loss':
                    best_loss[epoch - 1] = mean_value
                metrics_mean[key] = mean_value

                summary_writer.add_scalar(
                    scalar_value=mean_value,
                    tag=key,
                    global_step=epoch
                )

        summary_writer.add_hparams(
            hparam_dict=args.__dict__,
            metric_dict=metrics_mean,
            name=args.run_name,
            global_step=epoch
        )
        summary_writer.flush()

summary_writer.add_text('best_acc', f'Best accuracy: {np.max(best_acc)}')
summary_writer.add_text('best_loss', f'Best loss: {np.min(best_loss)}')