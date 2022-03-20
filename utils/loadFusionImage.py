'''
The function is to load image into a fusion. Returns: fusion image and image inforation (origin, spacing, and direction)
Default modalities: flair, t1, t1ce, and t2


Author: Linmin Pei
Date: Nov. 23th, 2020
'''

import os
import numpy as np
import SimpleITK as sitk


def loadFusionImage(sPath, pid, modality, bNorm=False, nLowPercentile=0.2):
    origin = []
    spacing = []
    direction = []
    imgData = []
    for idx, mode in enumerate(modality):
        imgPath = os.path.join(sPath, pid+mode)
        img_obj = sitk.ReadImage(imgPath)
        temp_origin = img_obj.GetOrigin()
        temp_spacing = img_obj.GetSpacing()
        temp_direction = img_obj.GetDirection()
        if len(origin) == 0:
            origin = temp_origin
        elif origin != temp_origin:
            raise ValueError('Error, origins are inconsistant')

        if len(spacing) == 0:
            spacing = temp_spacing
        elif spacing != temp_spacing:
            raise ValueError('Error, spacing are inconsistant')

        if len(direction) == 0:
            direction = temp_direction
        elif direction != temp_direction:
            raise ValueError('Error, direction are inconsistant')

        img = sitk.GetArrayFromImage(img_obj)
        img = img.astype(np.float64)

        if np.logical_and(bNorm == True, mode != '_seg.nii.gz'):
            mask = img > 0
            y = img[mask]
            lower = np.percentile(y, nLowPercentile)
            upper = np.percentile(y, 100-nLowPercentile)

            img[mask & (img < lower)] = lower
            img[mask & (img > upper)] = upper

            y = img[mask]

            img -= y.mean()
            img /= y.std()

        if len(imgData) == 0:
            img_shape = list(img.shape)  # convert shape tuple into a list
            img_shape.insert(0, len(modality))
            imgData = np.zeros(img_shape)
        imgData[idx] = img
    dataInfo = {'origin': origin,
                'spacing': spacing, 'direction': direction}

    return dataInfo, imgData
