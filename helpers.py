import os
import random
import pandas as pd
import torch
import math
import numpy as np
from torch.utils.data import Dataset


def get_data_full_path(path):
    train_data_path = os.path.join(path, 'train.csv')
    val_data_path = os.path.join(path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(path, 'test.csv')
    result_data_path = os.path.join(path, 'sample_submission.csv')
    return train_data_path, val_data_path, test_data_path, result_data_path


def preprocess(train_data_path, val_data_path, test_data_path):
    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)
    train_x = train.iloc[:, 1:].values / 255.
    train_y = train.iloc[:, 0].values
    train_mean, train_std = np.nanmean(train_x), np.nanstd(train_x) # normalisation
    train_x = (train_x - train_mean) / train_std
    train_x = np.reshape(train_x, (60000, 1, 28, 28))

    val_x = val.iloc[:, 1:].values / 255.
    val_y = val.iloc[:, 0].values
    val_mean, val_std = np.nanmean(val_x), np.nanstd(val_x)
    val_x = (val_x - val_mean) / val_std
    val_x = np.reshape(val_x, (10240, 1, 28, 28))

    test_x = test.iloc[:, 1:].values / 255.
    test_mean, test_std = np.nanmean(test_x), np.nanstd(test_x)
    test_x = (test_x - test_mean) / test_std
    test_x = np.reshape(test_x, (5000, 1, 28, 28))
    test_y = np.zeros(np.shape(test_x)[0])

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, batch, batch_test):

    torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to('cuda')
    torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor).to('cuda')
    train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)

    torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor).to('cuda')
    torch_val_y = torch.from_numpy(val_y).type(torch.LongTensor).to('cuda')
    val_dataset = torch.utils.data.TensorDataset(torch_val_x, torch_val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False, drop_last=False)

    torch_test_x = torch.from_numpy(test_x).type(torch.FloatTensor).to('cuda')
    torch_test_y = torch.from_numpy(test_y).type(torch.LongTensor).to('cuda')
    test_dataset = torch.utils.data.TensorDataset(torch_test_x, torch_test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_test, shuffle=True, drop_last=True)
    return train_loader, val_loader, test_loader





