import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Compose
# from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import *
# from .transforms import Pad, RandomCrop3D, RandomRotation, RandomFlip, RandomIntensityChange

# from .transforms import NumpyType


import numpy as np


class DriveData(Dataset):

    def __init__(self, root='', img_dir='', list_file='', trans=None, s_modality='f1c2'):
        nameList = []
        self.img_dir = os.path.join(root, img_dir)
        self.s_modality = s_modality
        self.n_channel = len(s_modality)
        self.diff_index = self.modality2index(s_modality)
        list_path = os.path.join(root, list_file)
        with open(list_path) as f:
            for name in f:
                nameList.append(name.split()[0])
        self.transforms = eval(trans)
        self.nameList = nameList

    def modality2index(self, s_modality, all_modality='f1c2'):
        assert len(all_modality) >= len(s_modality)
        dic = {'f': 0, '1': 1, 'c': 2, '2': 3}
        all_index = [dic[x] for x in all_modality]
        has_index = [dic[x] for x in s_modality]

        diff_index = list(set(all_index) - set(has_index))

        return diff_index

    def __getitem__(self, index):
        info_itk = {}
        img_path = os.path.join(self.img_dir, self.nameList[index]+'.nii.gz')
        imgData_obj = sitk.ReadImage(img_path)
        imgData = sitk.GetArrayFromImage(imgData_obj)  # 155x240x240x7
        imgData = np.transpose(imgData, (3, 0, 1, 2))  # 7x155x240x240
        # original sequence: fl, t1, t1ce, t2, necrosis, tumor core, and whole tumor (in subregion)
        imgData = np.delete(imgData, self.diff_index, 0)        

        temp_img, temp_mask = imgData[:self.n_channel], imgData[self.n_channel:]
        if temp_img.ndim == temp_mask.ndim+1:
            # ensure the temp_mask has same dimensionality as temp_img
            temp_mask = temp_mask[None, ...]
        # make a list: [img, label]
        imgFusion = [temp_img, temp_mask]
        if self.transforms is not None:
            imgFusion = self.transforms(imgFusion)
        # image/mask shape: 4/3x155x240x240
        image, mask = imgFusion[0], imgFusion[1]
        # image/mask shape: 1x4/3x155x240x240
        image, mask = image[None, ...], mask[None, ...]
        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return len(self.nameList)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
