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
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-dataset', default='DTD', type=str)
parser.add_argument('-loss', default='CCE', type=str)
parser.add_argument('-pdf', default='SOFT', type=str)
parser.add_argument('-gamma', default=1, type=float)
parser.add_argument('--local_rank', default=0, type=int)

args = parser.parse_args()

MAX_LEN = 0
if not torch.cuda.is_available():
    args.device = 'cpu'
    MAX_LEN = 200

if args.dataset == 'FASHION':
    class DatasetFashionMNIST(torch.utils.data.Dataset):
        def __init__(self, is_train):
            super().__init__()
            self.data = torchvision.datasets.FashionMNIST(
                root='../data',
                transform=torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                train=is_train,
                download=True
            )

            self.input_shape = (1, 28, 28)
            self.class_count = 10
            self.param_num = 8192

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
            self.data = torchvision.datasets.CIFAR10(
                root='../data',
                train=is_train,
                download=True
            )

            self.input_shape = (3, 32, 32)
            self.class_count = 10
            self.param_num = 8192

        def __len__(self):
            if MAX_LEN:
                return MAX_LEN
            return len(self.data)

        def __getitem__(self, idx):
            # list tuple np.array torch.FloatTensor
            pil_x, y_idx = self.data[idx]
            np_x = np.array(pil_x, dtype=np.float32) / 255.0
            np_x_new = np.empty([3, 32, 32])
            np_x_new[0, :, :] = np_x[:, :, 0]
            np_x_new[1, :, :] = np_x[:, :, 1]
            np_x_new[2, :, :] = np_x[:, :, 2]
            x = torch.FloatTensor(np_x_new)
            y = torch.LongTensor([y_idx])
            return x, y

    set_train = DatasetCIFAR(is_train=True)
    set_test = DatasetCIFAR(is_train=False)

elif args.dataset == "DTD":
    class DatasetDTD(torch.utils.data.Dataset):
        def __init__(self, is_train):
            super().__init__()
            self.data = torchvision.datasets.DTD(
                root='../data',
                split=is_train,
                transform=torch.nn.Sequential(
                    torchvision.transforms.Resize((300, 300)),
                    #torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET)
                ),
                download=True
            )

            self.input_shape = (3, 300, 300)
            self.class_count = 47
            self.param_num = 41472

        def __len__(self):
            if MAX_LEN:
                return MAX_LEN
            return len(self.data)

        def __getitem__(self, idx):
            # list tuple np.array torch.FloatTensor
            pil_x, y_idx = self.data[idx]
            np_x = np.array(pil_x, dtype=np.float32) / 255.0
            np_x_new = np.empty([3, 300, 300])
            np_x_new[0, :, :] = np_x[:, :, 0]
            np_x_new[1, :, :] = np_x[:, :, 1]
            np_x_new[2, :, :] = np_x[:, :, 2]
            x = torch.FloatTensor(np_x_new)
            y = torch.LongTensor([y_idx])
            return x, y

    set_train = DatasetDTD(is_train='train')
    set_test = DatasetDTD(is_train='test')


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


class EMDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, y, y_prim):
        self.loss = torch.sum(
            torch.log(1 / (y_prim[range(len(y)), y] + 1e-8))
        )
        return self.loss


class soft_margin_softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, dist):
        tens = torch.rand_like(logits)
        for sub in range(logits.size(dim=0)):
            for i in range(logits.size(dim=1)):
                tens[sub, i] = torch.exp(logits[sub, i] - dist) / (torch.exp(logits[sub, i] - dist) + torch.sum(torch.exp(logits[sub, :])) - torch.exp(logits[sub, i]))
        return tens


class taylor_softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        tens = torch.rand_like(logits)
        for sub in range(logits.size(dim=0)):
            for i in range(logits.size(dim=1)):
                tens[sub, i] = (1 + logits[sub, i] + torch.pow(logits[sub, i], 2) / 2) / (torch.sum(1 + logits[sub, :] + torch.pow(logits[sub, :], 2) / 2))
        return tens


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=set_train.param_num,
            out_features=set_train.class_count
        )

    def forward(self, x):
        batch_size = x.size(0)
        if args.dataset != 'DTD':
            x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(128, 128))  # upscale for pretrained deep model
        if args.dataset == 'FASHION':
            x = x.expand(x.size(0), 3, 128, 128) # repeat grayscale over RGB channels\
        out = self.encoder.forward(x)
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)
        #y_prim = soft_margin_softmax.forward(self, logits, 0.5) #soft-margin softmax
        #y_prim = taylor_softmax.forward(self, logits) #taylor softmax
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


for epoch in range(1, args.epochs + 1):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

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

            idx_y = np.array(y)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_mean = {}
        for key in metrics_epoch:
            if stage in key:
                mean_value = np.mean(metrics_epoch[key])
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