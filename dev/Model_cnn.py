import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def flatten(w=28, k=3, s=1, p=0):
    return int((np.floor((w - k + 2 * p) / s) + 1) / 1), k, s, p


def conv_layer(dim_in, dim_out, k, p, s, dropout=0.0):
    if dropout == 0.0 or dropout == 0:
        layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(dim_out),
            nn.PReLU()
        )
    else:
        layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=k, padding=p, stride=s),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(dim_out),
            nn.PReLU()
        )
    return layers


def fc_layer(dim_in, dropout=0.0):
    if dropout == 0.0 or dropout == 0:
        layers = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
    else:
        layers = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
    return layers


def proj_layer(dim_in, class_no):
    layer = nn.Linear(dim_in, class_no)
    return layer


class CNN(nn.Module):
    def __init__(self, n_classes=10, input_dim=1, width=32, depth=4, kernel=3, dropout_ratio=0.5):
        # A classic CNN
        super(CNN, self).__init__()

        if kernel == 3:
            padding = 1
        elif kernel == 5:
            padding = 2
        elif kernel == 7:
            padding = 3
        else:
            print('wrong kernel size')

        self.conv1 = conv_layer(dim_in=input_dim, dim_out=width, k=kernel, p=padding, s=2, dropout=dropout_ratio)
        r = flatten(w=28, k=kernel, s=2, p=padding)
        self.conv_layers = nn.ModuleList()

        # assert depth <= 5

        for i in range(depth-2):
            if i == 0:
                self.conv_layers.append(conv_layer(dim_in=width, dim_out=2*width, k=kernel, p=padding, s=2, dropout=dropout_ratio))
                r = flatten(w=r[0], k=kernel, s=2, p=padding)
            else:
                self.conv_layers.append(conv_layer(dim_in=2*width, dim_out=2*width, k=kernel, p=padding, s=1, dropout=dropout_ratio))
                r = flatten(w=r[0], k=kernel, s=1, p=padding)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc_layer = fc_layer(dim_in=r[0]*r[0]*width*2, dim_out=1024, dropout=0.5)
        self.output_layer = proj_layer(dim_in=1024, class_no=n_classes)

    def forward(self, x):
        x = self.conv1(x)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)

        x = self.flatten(x)
        x = self.fc_layer(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self, dropout=0.5):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)
        self.act4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act2(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act3(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = self.act4(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x