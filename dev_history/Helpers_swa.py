import os
import random
import pandas as pd
import torch
import math
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset

# Here includes data loader, data augmentation and a few small helper functions


def get_data_full_path(path):
    train_data_path = os.path.join(path, 'train.csv')
    val_data_path = os.path.join(path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(path, 'test.csv')
    return train_data_path, val_data_path, test_data_path


def get_data_values(path):
    train_data_path, val_data_path, test_data_path = get_data_full_path(path)
    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)

    # split the data:
    train_x = train.iloc[:, 1:].values / 255.
    train_y = train.iloc[:, 0].values

    val_x = val.iloc[:, 1:].values / 255.
    val_y = val.iloc[:, 0].values

    test_x = test.iloc[:, 1:].values / 255.
    test_id = test.iloc[:, 0].values

    # reshape the data:
    train_x = np.reshape(train_x, (60000, 1, 28, 28))
    val_x = np.reshape(val_x, (10240, 1, 28, 28))
    test_x = np.reshape(test_x, (5000, 1, 28, 28))

    return train_x, train_y, val_x, val_y, test_x, test_id


def calculate_mean_std(train_x, val_x, test_x):
    # calculate the mean and std:
    all_x = np.concatenate((train_x, val_x, test_x), axis=0)
    x_mean = np.nanmean(all_x)
    x_std = np.nanstd(all_x)
    return x_mean, x_std


class RandomZoom(object):
    # Zoom in augmentation
    # We zoom out the foreground parts when labels are available
    # We also zoom out the slices in the start and the end
    def __init__(self,
                 zoom_ratio=(0.7, 0.9)):
        self.zoom_ratio = zoom_ratio

    def sample_positions(self, image):
        ratio_h = round(random.uniform(self.zoom_ratio[0], self.zoom_ratio[1]), 2)
        ratio_w = round(random.uniform(self.zoom_ratio[0], self.zoom_ratio[1]), 2)
        # get image size upper bounds:
        h, w = np.shape(image)[-2], np.shape(image)[-1]
        # get cropping upper bounds:
        upper_h, upper_w = int(h*(1-ratio_h)), int(w*(1-ratio_w))
        # sampling positions:
        sample_h, sample_w = random.randint(0, upper_h), random.randint(0, upper_w)
        # sampling sizes:
        size_h, size_w = int(h * ratio_h), int(w * ratio_w)
        return sample_h, sample_w, size_h, size_w, ratio_h, ratio_w

    def sample_patch(self, image):
        h0, w0, new_h, new_w, ratio_h, ratio_w = self.sample_positions(image)
        cropped_image = image[:, h0:h0 + int(new_h), w0:w0 + int(new_w)]

        # upsample them:
        zoomed_image = scipy.ndimage.zoom(input=cropped_image, zoom=(1, math.ceil(1 / ratio_h), math.ceil(1 / ratio_w)), order=1)
        return zoomed_image

    def forward(self, image):
        image_zoomed = self.sample_patch(image)
        # crop again to makes the zoomed image has the same size as the original image size:
        h, w = np.shape(image)[-2], np.shape(image)[-1]
        image_zoomed = image_zoomed[:, 0:h, 0:w]
        return image_zoomed


class RandomContrast(object):
    def __init__(self, bin_range=(20, 255)):
        # self.bin_low = bin_range[0]
        # self.bin_high = bin_range[1]
        self.bin_range = bin_range

    def randomintensity(self, input):
        augmentation_flag = np.random.rand()
        if augmentation_flag >= 0.5:
            # bin = np.random.choice(self.bin_range)
            bin = random.randint(self.bin_range[0], self.bin_range[1])
            # c, d, h, w = np.shape(input)
            c, h, w = np.shape(input)
            image_histogram, bins = np.histogram(input.flatten(), bin, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            output = np.interp(input.flatten(), bins[:-1], cdf)
            output = np.reshape(output, (c, h, w))
        else:
            output = input
        return output


class RandomGaussian(object):
    def __init__(self, mean=0, std=0.01):
        self.m = mean
        self.sigma = std

    def gaussiannoise(self, input):
        noise = np.random.normal(self.m, self.sigma, input.shape)
        mask_overflow_upper = input + noise >= 1.0
        mask_overflow_lower = input + noise < 0.0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0.0
        input += noise
        return input


class RandomCut(object):
    def __init__(self, patch_range=(7, 14)):
        self.patch_range = patch_range

    def cutout(self, input):

        patch_h = np.random.randint(self.patch_range[0], self.patch_range[1])
        patch_w = np.random.randint(self.patch_range[0], self.patch_range[1])

        h0 = np.random.randint(0, 28 - patch_h)
        w0 = np.random.randint(0, 28 - patch_w)
        input[:, h0:h0+patch_h, w0:w0+patch_w] = 0

        return input


class DatasetKMNIST(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 augmentation_gaussian,
                 augmentation_contrast,
                 augmentation_zoom,
                 augmentation_cutout,
                 mean,
                 std):
        self.images = pd.read_csv(images_path)
        self.labels_path = labels_path
        if self.labels_path is not None:
            self.labels = pd.read_csv(labels_path)

        self.augmentation_gaussian = augmentation_gaussian
        self.augmentation_contrast = augmentation_contrast
        self.augmentation_zoom = augmentation_zoom
        self.augmentation_cutout = augmentation_cutout

        if self.augmentation_gaussian == 1:
            self.aug_gaussian_noise = RandomGaussian()
        if self.augmentation_contrast == 1:
            self.aug_contrast = RandomContrast(bin_range=(10, 255))
        if self.augmentation_zoom == 1:
            self.aug_random_zoom = RandomZoom()
        if self.augmentation_cutout == 1:
            self.aug_random_cutout = RandomCut()

        self.mean = mean
        self.std = std

    def __len__(self):
        return np.shape(self.images.iloc[:, 1:])[0]

    def __getitem__(self, index):
        # Read images:
        image = self.images.iloc[index, 1:].values.reshape((1, 28, 28)) / 255. # C x H x W

        # Do augmentation sequentially:
        if self.augmentation_gaussian == 1 and random.random() >= 0.5:
            image = self.aug_gaussian_noise.gaussiannoise(image)

        if self.augmentation_contrast == 1 and random.random() >= 0.5:
            image = self.aug_contrast.randomintensity(image)

        if self.augmentation_zoom == 1 and random.random() >= 0.5:
            image = self.aug_random_zoom.forward(image)

        if self.augmentation_cutout == 1 and random.random() >= 0.5:
            image = self.aug_random_cutout.cutout(image)

        # Normalisation on images:
        image = (image - self.mean + 1e-10) / (self.std + 1e-10)

        if self.labels_path is not None:
            label = self.labels.iloc[index, 0]
            return image, label
        else:
            return image


def train_swa(epochs, network, criterion, train_loader, optimizer, scheduler, val_loader, network_flag):

    swa_model = torch.optim.swa_utils.AveragedModel(network)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer=optimizer, swa_lr=0.05, anneal_epochs=5, anneal_strategy='cos')

    for j in range(epochs):

        network.train()
        running_loss = 0.0
        train_acc = 0.0
        counter_t = 0

        for i, data in enumerate(train_loader, 0):
            counter_t += 1
            inputs, labels = data
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum().item()
            train_acc += correct / inputs.size()[0]

        if j > 150 and j % 5 == 0:
            swa_model.update_parameters(network)
            swa_scheduler.step()
        else:
            scheduler.step()

        train_acc = train_acc / counter_t

        # evaluation:
        network.eval()
        val_acc = 0
        counter_v = 0

        with torch.no_grad():
            for data, target in val_loader:
                counter_v += 1
                data, target = data.to('cuda'), target.to('cuda')
                output = network(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc += correct / data.size()[0]
        val_acc = 100 * val_acc / counter_v
        print('network %d [epoch %d] loss: %.4f, train acc:% 4f, val acc: %.4f' % (network_flag, j + 1, running_loss / counter_t, train_acc, val_acc))

    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    network = swa_model

    # evaluation:
    network.eval()
    val_acc = 0
    counter_v = 0
    with torch.no_grad():
        for data, target in val_loader:
            counter_v += 1
            data, target = data.to('cuda'), target.to('cuda')
            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            val_acc += correct / data.size()[0]

    val_acc = 100 * val_acc / counter_v
    print('swa network %d val acc: %.4f' % (network_flag, val_acc))
    print('Finished Training of network %d' % network_flag)
    print('\n')
    return val_acc, network