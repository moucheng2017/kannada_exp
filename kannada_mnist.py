import pandas as pd
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os


def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to dir holding the dataset')
    parser.add_argument('inputlung', type=str, help='path to dir holding the dataset')
    parser.add_argument('--inputairway', '-a', default=None, type=str, help='path to dir holding the dataset')
    parser.add_argument('--outputdirname', '-o', default='output', type=str, help='path to dir to save the resultant')
    parser.add_argument('--savepreview', '-p', action='store_true', help='save preview for each case.')
    return parser


def main(args):
    # read the files
    train_data_path = os.path.join(args.data_path, 'train.csv')
    val_data_path = os.path.join(args.data_path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(args.data_path, 'test.csv')

    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)

    # split the data:
    train_x = train_data.iloc[:, 1:]
    train_y = train_data.iloc[:, 0]
    val_x = val_data.iloc[:, 1:]
    val_y = val_data.iloc[:, 0]
    test_x = test_data.iloc[:, 1:]
    test_id = test_data.iloc[:, 0]

    # reshape the data:
    train_x = np.reshape(train_x, (60000, 1, 28, 28))
    val_x = np.reshape(val_x, (10240, 1, 28, 28))
    test_x = np.reshape(test_x, (5000, 1, 28, 28))

    # train data loader:
    torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # val data loader
    torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
    torch_val_y = torch.from_numpy(val_y).type(torch.LongTensor)
    val_dataset = torch.utils.data.TensorDataset(torch_val_x, torch_val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # define a model


    # train the network with validation

    # test the network with testing data

    test_result = 0

    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'resize dataset into 512x512x-1, original source, awy and lung masks must all be .nii.gz',
        parents=[args_parser()])
    args = parser.parse_args()
    main(args)