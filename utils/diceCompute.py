'''the function is to compute the tumor overlap dice score
!!!!!!!!!!!!!The input is a torch Variable Cuda data type, not numpy data. Otherwise it won't work.
#usage: img----->image
#       gt------>ground truth in array
#       losstype-->0-->complete tumor dice loss, 1-->tumor core loss, 2--> enhancing tumor loss only. default as 0
# return: DSC
'''
import numpy as np


def dice_binary(img_bin, gt_bin):
    inter_bin = img_bin*gt_bin
    if np.sum(img_bin)+np.sum(gt_bin) == 0:
        sub_dice = 1.0
    else:
        sub_dice = 2.0*np.sum(inter_bin)/(np.sum(img_bin)+np.sum(gt_bin))
    return sub_dice


def diceCompute(img, gt):
    assert img.shape == gt.shape

    img = img.flatten()  # convert to 1D
    gt = gt.flatten()  # convert to 1D

    img_wt = (img > 0).astype(int)
    gt_wt = (gt > 0).astype(int)
    dice_wt = dice_binary(img_wt, gt_wt)


    return dice_wt
