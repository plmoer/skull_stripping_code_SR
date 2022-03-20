import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


def diceloss(output, target, eps=1e-5):  # soft dice loss
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def bceloss(output, target):
    bceObj = nn.BCELoss()
    loss = bceObj(output, target)
    return loss


# Generalized dice loss
def weightedloss(output, target, eps=1e-5, weight_type='square'):
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """
    output = torch.transpose(output, 0, 1)  # nB x nC x nSlice x nRow x nColumn
    target = torch.transpose(target, 0, 1)  # nB x nC x nSlice x nRow x nColumn

    # nC, nB x nSlice x nRow x nColumn
    target = torch.flatten(target, start_dim=1)
    output = torch.flatten(output, start_dim=1)

    target_sum = target.sum(-1)

    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(
        loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum
