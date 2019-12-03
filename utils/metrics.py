import torch
from torch.nn import functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = intersection.item() / union.item()
        thresholds = torch.arange(0.5, 1, 0.05)
        iou_th = []
        for thresh in thresholds:
            iou_th.append(iou > thresh)

        ious.append(np.mean(iou_th))

    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


def main_dice_metric(logits, true, eps=1e-7, do_sigmoid_or_softmax=True):
    true = true.unsqueeze(0)
    logits = logits.unsqueeze(0)
    # print(true.shape, logits.shape)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        if do_sigmoid_or_softmax:
            pos_prob = torch.sigmoid(logits)
        else:
            pos_prob = logits
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        if do_sigmoid_or_softmax:
            probas = F.softmax(logits, dim=1)
        else:
            probas = logits
    true_1_hot = true_1_hot.type(logits.type())
    dims = tuple(range(2, true.ndimension()))
    # print("#", probas.shape, true_1_hot.shape)
    intersection = probas * true_1_hot
    intersection = torch.sum(intersection, dims)
    # print("Inter shape", intersection.shape)
    cardinality = probas + true_1_hot
    # print("Card shape", cardinality.shape)
    # print("Dims", dims)
    cardinality = torch.sum(cardinality, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean(dim=1)
    # print(dice_loss)
    return dice_loss.item()


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
