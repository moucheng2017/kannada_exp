import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import os


def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', '-s', default=2022, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.9, type=float, help='exponential learning decay rate')
    parser.add_argument('--step_size', default=0.7, type=float, help='step size')

    parser.add_argument('--epochs', '-e', default=20, type=int, help='Training epochs')
    # parser.add_argument('--depth', '-d', default=18, type=int, help='Depth of Network')
    # parser.add_argument('--width', '-w', default=32, type=int, help='Width of Network')
    parser.add_argument('--path', '-p', default=None, type=str, help='Path to the folder containing all of the data in csv')
    parser.add_argument('--device', '-de', default='gpu', type=str, help='device choice between gpu and cpu')
    return parser


def main(args):
    # read the files
    train_data_path = os.path.join(args.path, 'train.csv')
    val_data_path = os.path.join(args.path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(args.path, 'test.csv')

    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)

    # split the data:
    train_x = train.iloc[:, 1:].values / 255.
    # train_x = (train_x - 0.1307) / 0.3081
    train_y = train.iloc[:, 0].values

    val_x = val.iloc[:, 1:].values / 255.
    # val_x = (val_x - 0.1307) / 0.3081
    val_y = val.iloc[:, 0].values

    test_x = test.iloc[:, 1:].values / 255.
    # test_x = (test_x - 0.1307) / 0.3081
    test_id = test.iloc[:, 0].values

    # reshape the data:
    train_x = np.reshape(train_x, (60000, 1, 28, 28))
    val_x = np.reshape(val_x, (10240, 1, 28, 28))
    test_x = np.reshape(test_x, (5000, 1, 28, 28))

    # calculate the mean and std:
    all_x = np.concatenate((train_x, val_x, test_x), axis=0)
    x_mean = np.nanmean(all_x)
    x_std = np.nanstd(all_x)

    # normalise images:
    train_x = (train_x - x_mean + 1e-8) / (x_std + 1e-8)
    val_x = (val_x - x_mean + 1e-8) / (x_std + 1e-8)
    test_x = (test_x - x_mean + 1e-8) / (x_std + 1e-8)

    # train data loader:
    if args.device == 'gpu':
        torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to('cuda')
        torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor).to('cuda')
    else:
        torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # val data loader
    if args.device == 'gpu':
        torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor).to('cuda')
        torch_val_y = torch.from_numpy(val_y).type(torch.FloatTensor).to('cuda')
    else:
        torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
        torch_val_y = torch.from_numpy(val_y).type(torch.FloatTensor)

    val_dataset = torch.utils.data.TensorDataset(torch_val_x, torch_val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, downsample):
            super().__init__()
            if downsample:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                self.shortcut = nn.Sequential()

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, input):
            shortcut = self.shortcut(input)
            input = nn.ReLU()(self.bn1(self.conv1(input)))
            input = nn.ReLU()(self.bn2(self.conv2(input)))
            input = input + shortcut
            return nn.ReLU()(input)

    class ResNet(nn.Module):
        def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
            super().__init__()
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

            if useBottleneck:
                filters = [64, 256, 512, 1024, 2048]
            else:
                filters = [64, 64, 128, 256, 512]

            self.layer1 = nn.Sequential()
            self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
            for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d' % (i + 1,), resblock(filters[1], filters[1], downsample=False))

            self.layer2 = nn.Sequential()
            self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
            for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i + 1,), resblock(filters[2], filters[2], downsample=False))

            self.layer3 = nn.Sequential()
            self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
            for i in range(1, repeat[2]):
                self.layer3.add_module('conv2_%d' % (i + 1,), resblock(filters[3], filters[3], downsample=False))

            self.layer4 = nn.Sequential()
            self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
            for i in range(1, repeat[3]):
                self.layer4.add_module('conv3_%d' % (i + 1,), resblock(filters[4], filters[4], downsample=False))

            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(filters[4], outputs)

        def forward(self, input):
            input = self.layer0(input)
            input = self.layer1(input)
            input = self.layer2(input)
            input = self.layer3(input)
            input = self.layer4(input)
            input = self.gap(input)
            input = torch.flatten(input, start_dim=1)
            input = self.fc(input)

            return input

    net = ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=10).to('cuda')

    # define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # train the network with validation
    torch.manual_seed(args.seed)

    # train the network
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        counter_t = 0
        for i, data in enumerate(train_loader, 0):
            counter_t += 1
            inputs, labels = data
            #inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum().item()
            train_acc += correct / inputs.size()[0]
        # print('[epoch %d] loss: %.3f' % (epoch + 1, running_loss / sampling_times))
        scheduler.step()
        train_acc = train_acc / counter_t

        # evaluation:
        net.eval()
        # correct = 0
        acc = 0
        counter_v = 0
        with torch.no_grad():
            for data, target in val_loader:
                counter_v += 1
                data, target = data.to('cuda'), target.to('cuda')
                # print(data.size())
                output = net(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc += correct / data.size()[0]
                # print('Accuracy of the network %d %%' % acc)
        # val_acc = 100 * correct / len(val_dataset)
        val_acc = 100 * acc / counter_v
        # print('Accuracy of the network %d %%' % val_acc)
        print('[epoch %d] loss: %.4f, train acc:% 4f, val acc: %.4f' % (epoch + 1, running_loss / counter_t, train_acc, val_acc))

    print('Finished Training\n')

    # validating
    # net.eval()
    # val = net(torch_val_x)
    # _, predicted = torch.max(val.data, 1)
    # # acc = 100 * torch.sum(torch_val_y == predicted) / len(torch_val_y)
    # # print('Accuracy of the network %d %%' % acc)
    # print('Accuracy of the network %d %%' % (100 * torch.sum(torch_val_y == predicted) / len(val_y)))

    # net.eval()
    # # correct = 0
    # acc = 0
    # counter = 0
    # with torch.no_grad():
    #     for data, target in val_loader:
    #         counter += 1
    #         data, target = data.to('cuda'), target.to('cuda')
    #         # print(data.size())
    #         output = net(data)
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct = pred.eq(target.view_as(pred)).sum().item()
    #         acc += correct / data.size()[0]
    #         # print('Accuracy of the network %d %%' % acc)
    #
    # # val_acc = 100 * correct / len(val_dataset)
    # val_acc = 100 * acc / counter
    # print('Accuracy of the network %d %%' % val_acc)

    return net


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # change the hyper parameters here:
    args.path = '/home/moucheng/projects_data/Kannada/Kannada-MNIST'
    args.batch = 1024
    args.device = 'gpu'
    args.epochs = 100
    args.seed = 1234
    args.lr = 0.001
    args.gamma = 0.8
    args.step_size = 10

    main(args)