import pandas as pd
import numpy as np
import example
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

    # define a model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
            self.fc1 = nn.Linear(3 * 3 * 64, 128)
            self.fc2 = nn.Linear(128, 10)

            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
            self.act3 = nn.ReLU(inplace=True)
            self.act4 = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.act1(self.conv1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.act2(F.max_pool2d(self.conv2(x), 2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.act3(F.max_pool2d(self.conv3(x), 2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.view(-1, 3 * 3 * 64)
            x = self.act4(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc2(x)
            # return x
            return F.log_softmax(x, dim=1)

    if args.device == 'gpu':
        net = Net().to('cuda')
    else:
        net = Net()

    # define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=0.01)
    # optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    # train the network with validation
    torch.manual_seed(args.seed)

    # train the network
    net.train()
    for epoch in range(args.epochs):
        # running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #inputs, labels = inputs, labels

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print out stats
            if (i+1) == (len(train_dataset) // args.batch):
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                print('[epoch %d] loss: %.3f' % (epoch + 1, loss.item()))
                # print('[%d, %5d] loss: %.3f, acc: %.3f' % (epoch + 1, i + 1, loss.item(), acc))

        # scheduler.step()

    print('Finished Training\n')

    # validating
    # net.eval()
    # val = net(torch_val_x)
    # _, predicted = torch.max(val.data, 1)
    # # acc = 100 * torch.sum(torch_val_y == predicted) / len(torch_val_y)
    # # print('Accuracy of the network %d %%' % acc)
    # print('Accuracy of the network %d %%' % (100 * torch.sum(torch_val_y == predicted) / len(val_y)))

    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_acc = 100 * correct / len(val_dataset)
    print('Accuracy of the network %d %%' % val_acc)

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
    args.gamma = 0.999

    main(args)