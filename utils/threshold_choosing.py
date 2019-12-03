"""
Module with functionality for choosing the best threshold on Validation Dataset
"""
from tqdm import tqdm
import numpy as np


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds + targs).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


def torch_dice_metric(y_pred, y_true, smooth = 1.0):
    y_pred = y_pred.float()
    y_true = y_true.float()
    assert y_pred.size() == y_true.size()
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (
        y_pred.sum() + y_true.sum() + smooth
    )
    return dsc


def choose_threshold(preds, targs):
    """
    Chooses and returns the best threshold for Test Dataset
    :param preds: tensor: [B, C, W, H] shape,
            where C is number of classes - 2 in our case.
            Tensor of data got from model after softmax.
    :param targs: tensor: [B, 1, W, H] shape of Labels, Y
    :return: (best_thrs, history): best threshold value and history list of pairs [thrs, dice_value]
    """
    dices = []
    thrs = np.arange(0.0, 1.01, 0.01)
    for i in tqdm(thrs):
        preds_m = (preds > i).long()
        dices.append(torch_dice_metric(preds_m, targs).mean())
    dices = np.array(dices)
    best_thrs = thrs[np.argmax(dices)]
    return best_thrs, list(zip(thrs.tolist(), dices.tolist()))
