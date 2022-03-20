'''
This function is to compute an average dice between segmentation and ground truth.
Note that both inputs are tensors.

Author: Linmin Pei
Date: Nov. 18th, 2020
'''
import numpy as np
from utils import mask_conversion
from .diceCompute import diceCompute


def cal_dice(pred, target, bGPU=False, nBatch_size=1, bTradition=False):
    # only works for batch size as 1
    assert nBatch_size == 1
    maskConversionObj = getattr(mask_conversion, 'de_conversion')

    if bGPU == True:
        target_arr = target.cpu().detach().numpy()  # tensor-->numpy array
        pred_arr = pred.cpu().detach().numpy()  # tensor-->numpy array
    else:
        target_arr = target.detach().numpy()  # tensor-->numpy array
        pred_arr = pred.detach().numpy()  # tensor-->numpy array
    target_arr = np.squeeze(target_arr)
    pred_arr = np.squeeze(pred_arr)
    target_arr = (target_arr > 0.5).astype(np.int)
    pred_arr = (pred_arr > 0.5).astype(np.int)

    # target_label = maskConversionObj(target_arr, bTradition)
    # pred_label = maskConversionObj(pred_arr, bTradition)
    dice = diceCompute(pred_arr, target_arr)
    return dice
