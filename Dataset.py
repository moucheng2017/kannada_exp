import os
import random
import pandas as pd
import torch
import math
import numpy as np
import scipy.ndimage
from skimage.transform import resize
import numpy.ma as ma
from torch.utils.data import Dataset


class RandomZoom(object):
    # Zoom in augmentation
    # We zoom out the foreground parts when labels are available
    # We also zoom out the slices in the start and the end
    def __init__(self,
                 zoom_ratio=(0.5, 0.9)):
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
    def __init__(self, bin_range=(100, 255)):
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


class DatasetKannadaMNIST(Dataset):
    def __init__(self, images_path, labels_path, augmentation, mean, std):
        self.images = pd.read_csv(images_path)
        self.labels_path = labels_path
        if self.labels_path is not None:
            self.labels = pd.read_csv(labels_path)

        self.augmentation = augmentation
        if self.augmentation == 1:
            self.aug_gaussian_noise = RandomGaussian()
            self.aug_contrast = RandomContrast(bin_range=(10, 255))
            self.aug_random_zoom = RandomZoom()
            self.aug_random_cutout = RandomCut()

        self.mean = mean
        self.std = std

    def __len__(self):
        return np.shape(self.images.iloc[:, 1:])[0]

    def __getitem__(self, index):
        # Read images:
        image = self.images.iloc[index, 1:].values.reshape((1, 28, 28)) / 255. # C x H x W

        # Normlisation on image:
        image = (image - self.mean + 1e-10) / (self.std + 1e-10)

        if self.augmentation == 1:
            # do augmentation
            if random.random() >= 0.5:
                image = self.aug_gaussian_noise.gaussiannoise(image)
                # image = self.aug_contrast.randomintensity(image)
                # image = self.aug_random_zoom.forward(image)
                image = self.aug_random_cutout.cutout(image)
                image = (image - self.mean + 1e-10) / (self.std + 1e-10)

        if self.labels_path is not None:
            label = self.labels.iloc[index, 0]
            return image, label
        else:
            return image


# class DatasetKMNIST(Dataset):
#     def __init__(self, images_path, labels_path, transform):
#         self.images = pd.read_csv(images_path)
#         self.labels_path = labels_path
#         self.transform = transform
#         if self.labels_path is not None:
#             self.labels = pd.read_csv(labels_path)
#
#     def __len__(self):
#         return np.shape(self.images.iloc[:, 1:])[0]
#
#     def __getitem__(self, index):
#         # Read images:
#         image = self.images.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1)) # H x W x C
#         image = self.transform(image)
#         # Read labels:
#         if self.labels_path is not None:
#             label = self.labels.iloc[index, 0]
#             return image, label
#         else:
#             return image







