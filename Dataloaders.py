import os
import torch
import random
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


class KMNIST_Data(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder,
                 new_size_h=384,
                 new_size_w=384,
                 full_orthogonal=0,
                 sampling_weight=5,
                 lung_window=1,
                 normalisation=1,
                 gaussian_aug=1,
                 zoom_aug=1,
                 contrast_aug=1):

        # flags
        # self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.normalisation_flag = normalisation
        self.zoom_aug_flag = zoom_aug

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        self.lung_window_flag = lung_window

        if self.contrast_aug_flag == 1:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag == 1:
            self.gaussian_noise = RandomGaussian()

    def __getitem__(self, index):
        # Lung masks:
        # all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        # lung = nib.load(all_lungs[index])
        # lung = lung.get_fdata()
        # lung = np.array(lung, dtype='float32')
        # lung = np.transpose(lung, (2, 0, 1))

        # Check image extension:
        image_example = os.listdir(self.imgs_folder)[0]
        if image_example.lower().endswith(('.nii.gz', '.nii')):
            # Images:
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
            imagename = all_images[index]
            # load image and preprocessing:
            image = nib.load(imagename)
            image = image.get_fdata()

        else:
            # Images:
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
            imagename = all_images[index]
            # load image and preprocessing:
            image = np.load(imagename)

        image = np.array(image, dtype='float32')
        # transform dimension:
        image = np.transpose(image, (2, 0, 1)) # (H x W x D) --> (D x H x W)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # Now applying lung window:
        if self.lung_window_flag == 1:
            image[image < -1000.0] = -1000.0
            image[image > 500.0] = 500.0

        if self.lbls_folder:
            # Labels:
            label_example = os.listdir(self.lbls_folder)[0]
            if label_example.lower().endswith(('.nii.gz', '.nii')):
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
                label = nib.load(all_labels[index])
                label = label.get_fdata()
            else:
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.npy*')))
                label = np.load(all_labels[index])

            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))

            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            if self.normalisation_flag == 1:
                image = normalisation(label, image)

            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag == 1:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag == 1:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # Apply normalisation at each case-wise again:
            if self.normalisation_flag == 1:
                image_weighted = normalisation(label, image_weighted)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image_weighted, label)

            return inputs_dict, imagename

        else:
            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            if self.normalisation_flag == 1:
                image = normalisation(None, image)
            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag == 1:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag == 1:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # Apply normalisation at each case-wise again:
            if self.normalisation_flag == 1:
                image_weighted = normalisation(None, image_weighted)

            inputs_dict = self.augmentation_cropping.crop(image_weighted)

            return inputs_dict, imagename

    def __len__(self):
        example = os.listdir(self.imgs_folder)[0]
        if example.lower().endswith(('.nii.gz', '.nii')):
            # You should change 0 to the total size of your dataset.
            return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        elif example.lower().endswith('.npy'):
            return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))