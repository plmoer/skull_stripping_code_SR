'''
transformation function
Author: Linmin Pei
Modified date: Nov. 15th, 2020
'''
import random
import collections
import numpy as np
import torch
from scipy import ndimage
import SimpleITK as sitk

from .rand import Constant, Uniform, Gaussian
from scipy.ndimage import rotate

# imgList: [img, mask], where img.shape=4x155x240x240
# and mask.shape=4x155x240x240 if use subregion


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgList):
        for t in self.transforms:
            imgList = t(imgList)
        return imgList


class CenterCrop3D(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgList):
        new_imgList = []
        if len(imgList) > 1:
            shape = imgList[0][0].shape  # 155x240x240
        else:
            raise ValueError('Something is wrong on image data')

        # check if shape is greater than size
        bValue = all(x <= y for x, y in zip(self.size, shape))
        if bValue == True:
            # get center start position
            start = [(s-i)//2 for i, s in zip(self.size, shape)]
            # start=[71, 36, 24]
            buffer = [slice(s, s+k) for s, k in zip(start, self.size)]
            # buffer=[slice(71, 199, None), slice(36, 164, None), slice(24, 152, None)]
            for idx in range(len(imgList)):
                temp = [imgList[idx][c][tuple(buffer)]
                        for c in range(len(imgList[idx]))]  # crop image
                new_img = np.stack(temp, axis=0)
                new_imgList.append(new_img)  # make a
        else:
            raise ValueError('image shape must be greater than defined size')

        return new_imgList


class RandomCrop3D(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgList):
        new_imgList = []
        if len(imgList) > 1:
            shape = imgList[0][0].shape  # 155x240x240
        else:
            raise ValueError('Something is wrong on image data')

        # check if shape is greater than size
        bValue = all(x <= y for x, y in zip(self.size, shape))
        if bValue == True:
            # get random start position
            start = [random.randint(0, s-i) for i, s in zip(self.size, shape)]
            # start=[71, 36, 24]
            buffer = [slice(s, s+k) for s, k in zip(start, self.size)]
            # buffer=[slice(71, 199, None), slice(36, 164, None), slice(24, 152, None)]
            for idx in range(len(imgList)):
                temp = [imgList[idx][c][tuple(buffer)]
                        for c in range(len(imgList[idx]))]  # crop image
                new_img = np.stack(temp, axis=0)
                new_imgList.append(new_img)  # make a
        else:
            raise ValueError('image shape must be greater than defined size')

        return new_imgList


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle
        axes = [(1, 0), (2, 1), (2, 0)]
        self.axes = axes

    def __call__(self, imgList):
        new_imgList = []
        nPadValue = -3  # padding value for image
        # imgList: [img, mask]
        if len(imgList) < 1:
            raise ValueError('Something is wrong on image data')
        # generate a random axis
        axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))]
        # generate a random angle
        angle_buffer = np.random.randint(-self.angle, self.angle)
        for idx in range(len(imgList)):
            if idx == 1:
                nPadValue = 0  # padding value for mask
            temp = [rotate(imgList[idx][c], angle_buffer, axes=axes_buffer, reshape=False,
                           order=0, mode='constant', cval=nPadValue) for c in range(len(imgList[idx]))]  # rotate image
            new_img = np.stack(temp, axis=0)
            new_imgList.append(new_img)  # make a list: [image, mask]

        return new_imgList


class RandomIntensityChange(object):
    def __init__(self, factor):
        shift, scale = factor
        assert (shift > 0) and (scale > 0)
        self.shift = shift
        self.scale = scale

    def __call__(self, imgList):
        new_imgList = []
        if len(imgList) < 1:
            raise ValueError('Something is wrong on image data')

        shift_factor = np.random.uniform(-self.shift, self.shift,
                                         size=[imgList[0].shape[0], imgList[0].shape[1], 1, 1])
        scale_factor = np.random.uniform(
            1.0 - self.scale, 1.0 + self.scale, size=[imgList[0].shape[0], imgList[0].shape[1], 1, 1])
        img_shift = imgList[0] * scale_factor + shift_factor
        new_imgList = [img_shift, imgList[1]]

        return new_imgList


class RandomFlip(object):
    def __init__(self):
        self.axis = (1, 2, 3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def __call__(self, imgList):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        # print('----333 random flip: ', self.x_buffer,
        #   ' y: ', self.y_buffer, ' z: ', self.z_buffer)
        if len(imgList) < 1:
            raise ValueError('Something is wrong on image data')

        for idx in range(len(imgList)):
            if self.x_buffer:
                imgList[idx] = np.flip(imgList[idx], axis=self.axis[0]).copy()
            if self.y_buffer:
                imgList[idx] = np.flip(imgList[idx], axis=self.axis[1]).copy()
            if self.z_buffer:
                imgList[idx] = np.flip(imgList[idx], axis=self.axis[2]).copy()

        return imgList


class Pad(object):
    def __init__(self, size):
        self.size = size
        self.px = tuple(zip([0]*len(size), size))

    def __call__(self, imgList):
        for idx in range(len(imgList)):
            if len(imgList[idx]) > 1:  # to avoid empty if no ground truth provided
                dim = len(imgList[idx])
                # after normalization, pix value in background is around -3
                imgList[idx] = np.pad(imgList[idx], self.px[:dim],
                                      mode='constant', constant_values=-3)
        return imgList
