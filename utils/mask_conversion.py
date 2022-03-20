'''
The class is to do mask conversion/deconversion, specially for subregion.
For example, in brain tumor, it has edema (ED), necrosis (NC), and enhancing tumor (ET). We might to 
use tumor subregion, usch as tumor core (TC), whole tumor (WT), or ET. TC=NC+ET, WT=ED+NC+ET.

Author: Linmin Pei
Date: Nov. 18th, 2020
'''
import numpy as np


def gt_conversion(gt, nClass=3, bTradition=False):
    # gt: 155x240x240--->3x155x240x240

    # convert original gt to subregion form
    gt_shape = list(gt.shape)  # 155x240x240
    gt_shape.insert(0, nClass)  # 3x155x240x240
    new_gt = np.zeros(gt_shape)

    temp = np.zeros(gt.shape)  # temp image: 155x240x240
    if bTradition == False:
        temp[np.where(gt == 1)] = 1
    else:
        temp[np.where(gt == 4)] = 1
    new_gt[0] = temp  # for necrosis

    temp = np.zeros(gt.shape)  # temp image: 155x240x240
    temp[np.where(np.logical_or(gt == 1, gt == 4))] = 1
    new_gt[1] = temp  # for tumor core

    temp = np.zeros(gt.shape)  # temp image: 155x240x240
    temp[np.where(gt > 0)] = 1
    new_gt[2] = temp  # for whole tumor

    return new_gt


def de_conversion(gt, bTradition=False):
    # gt: 3x155x240x240--->155x240x240
    gt_shape = list(gt.shape)  # 3x155x240x240
    gt_shape.pop(0)  # 155x240x240
    new_gt = np.zeros(gt_shape)

    index_ed = np.where(gt[2] == 1)
    new_gt[index_ed] = 2  # edema

    index_nc = np.where(gt[1] == 1)
    index_et = np.where(gt[0] == 1)
    if bTradition == False:
        new_gt[index_nc] = 4  # enhancing tumor
        new_gt[index_et] = 1  # necrosis
    else:
        new_gt[index_nc] = 1  # necrosis
        new_gt[index_et] = 4  # enhancing tumor

    return new_gt
