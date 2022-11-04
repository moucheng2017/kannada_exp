import os
import random
import pandas as pd
import torch
import math
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset


def get_data_full_path(path):
    train_data_path = os.path.join(path, 'train.csv')
    val_data_path = os.path.join(path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(path, 'test.csv')
    result_data_path = os.path.join(path, 'sample_submission.csv')
    return train_data_path, val_data_path, test_data_path, result_data_path


class DatasetKMNIST(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 mean,
                 std):
        '''
        :param images_path:
        :param labels_path:
        :param mean:
        :param std:
        '''
        self.images = pd.read_csv(images_path)
        self.labels_path = labels_path
        if self.labels_path is not None:
            self.labels = pd.read_csv(labels_path)

        self.aug_strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)), # upsample it from 28 to 32 because vgg has 3 downsampling stages!
            transforms.RandomAffine(degrees=10, translate=(0.25, 0.25), scale=(0.75, 1.25), shear=0.1),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomResizedCrop((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.aug_weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return np.shape(self.images.iloc[:, 1:])[0]

    def __getitem__(self, index):
        # Read images:
        image = self.images.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))  # PIL needs H W C

        s_image = self.aug_strong(image)
        w_image = self.aug_weak(image)

        if self.labels_path is not None:
            label = self.labels.iloc[index, 0]
            return s_image.float(), w_image.float(), torch.tensor(label, dtype=torch.long)
        else:
            return s_image.float(), w_image.float()


