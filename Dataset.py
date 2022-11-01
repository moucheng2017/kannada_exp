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
                 zoom_ratio_h=(0.5, 0.7),
                 zoom_ratio_w=(0.5, 0.7)):

        self.zoom_ratio_h = zoom_ratio_h
        self.zoom_ratio_w = zoom_ratio_w

    def sample_positions(self, label):
        ratio_h = round(random.uniform(self.zoom_ratio_h[0], self.zoom_ratio_h[1]), 2)
        ratio_w = round(random.uniform(self.zoom_ratio_w[0], self.zoom_ratio_w[1]), 2)
        # get image size upper bounds:
        h, w = np.shape(label)[-2], np.shape(label)[-1]
        # get cropping upper bounds:
        upper_h, upper_w = int(h*(1-ratio_h)), int(w*(1-ratio_w))
        # sampling positions:
        sample_h, sample_w = random.randint(0, upper_h), random.randint(0, upper_w)
        # sampling sizes:
        size_h, size_w = int(h * ratio_h), int(w * ratio_w)
        return sample_h, sample_w, size_h, size_w, ratio_h, ratio_w

    def sample_patch(self, image, label):

        h0, w0, new_h, new_w, ratio_h, ratio_w = self.sample_positions(label)
        cropped_image = image[h0:h0 + int(new_h), w0:w0 + int(new_w)]
        cropped_label = label[h0:h0 + int(new_h), w0:w0 + int(new_w)]

        # upsample them:
        zoomed_image = scipy.ndimage.zoom(input=cropped_image, zoom=(math.ceil(1 / ratio_h), math.ceil(1 / ratio_w)), order=1)
        zoomed_label = scipy.ndimage.zoom(input=cropped_label, zoom=(math.ceil(1 / ratio_h), math.ceil(1 / ratio_w)), order=0)

        return zoomed_image, zoomed_label

    def forward(self, image, label):
        image, label = np.squeeze(image), np.squeeze(label)
        image_zoomed, label_zoomed = self.sample_patch(image, label)

        # crop again to makes the zoomed image has the same size as the original image size:
        h, w = np.shape(label)[-2], np.shape(label)[-1]
        image_zoomed, label_zoomed = image_zoomed[0:h, 0:w], label_zoomed[0:h, 0:w]
        h2, w2 = np.shape(label_zoomed)[-2], np.shape(label_zoomed)[-1]

        assert h2 == h
        assert w2 == w

        return image_zoomed, label_zoomed


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


def randomcutout(x, y):
    '''
    Args:
        x: segmentation
        y: gt
    Returns:
    '''
    b, c, h, w = x.size()
    h_mask, w_mask = random.randint(int(h // 5), int(h // 2)), random.randint(int(w // 5), int(w // 2))

    h_starting = np.random.randint(0, h - h_mask)
    w_starting = np.random.randint(0, w - h_mask)
    h_ending = h_starting + h_mask
    w_ending = w_starting + w_mask

    mask = torch.ones_like(x).cuda()
    mask[:, :, h_starting:h_ending, w_starting:w_ending] = 0

    return x*mask, y*mask


def normalisation(label, image):
    # Case-wise normalisation
    # Normalisation using values inside of the foreground mask

    if label is None:
        lung_mean = np.nanmean(image)
        lung_std = np.nanstd(image)
    else:
        image_masked = ma.masked_where(label > 0.5, image)
        lung_mean = np.nanmean(image_masked)
        lung_std = np.nanstd(image_masked)

    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


class DatasetMNIST(Dataset):

    def __init__(self, file_path, transform, mean, std):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        # image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)
            # image = image / 255.
            # image = (image - self.x_mean) / self.x_std

        return image, label

# class KMNIST_Data(Dataset):
#     def __init__(self,
#                  images_folder,
#                  labels_folder,
#                  normalisation_mean,
#                  normalisation_std,
#                  gaussian_aug=1,
#                  zoom_aug=1,
#                  contrast_aug=1):
#
#         # data
#         self.imgs_folder = images_folder
#         self.lbls_folder = labels_folder
#
#         self.augmentation_contrast = RandomContrast(bin_range=(20, 255))
#         self.gaussian_noise = RandomGaussian()
#
#     def __getitem__(self, index):
#         image = pd.read_csv(self.imgs_folder)
#         image = image.iloc[:, 1:].values / 255.
#         image = np.reshape(image, (-1, 1, 28, 28))
#         image = np.array(image, dtype='float32')
#
#         lbl = pd.read_csv(self.lbls_folder)
#         lbl = lbl.iloc[:, 1:].values / 255.
#         lbl = np.reshape(lbl, (-1, 1, 28, 28))
#
#         image = normalisation(label, image)
#
#         image_queue.append(image)
#
#         # Random contrast:
#         if self.contrast_aug_flag == 1:
#             image_another_contrast = self.augmentation_contrast.randomintensity(image)
#             image_queue.append(image_another_contrast)
#
#         # Random Gaussian:
#         if self.gaussian_aug_flag == 1:
#             image_noise = self.gaussian_noise.gaussiannoise(image)
#             image_queue.append(image_noise)
#
#         # weights:
#         dirichlet_alpha = collections.deque()
#         for i in range(len(image_queue)):
#             dirichlet_alpha.append(1)
#         dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)
#
#         # make a new image:
#         image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
#         image_weighted = sum(image_weighted)
#
#         # Apply normalisation at each case-wise again:
#         if self.normalisation_flag == 1:
#             image_weighted = normalisation(label, image_weighted)
#
#         # get slices by weighted sampling on each axis with zoom in augmentation:
#         inputs_dict = self.augmentation_cropping.crop(image_weighted, label)
#
#         return inputs_dict, imagename

