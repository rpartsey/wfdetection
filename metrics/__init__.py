import torch


import numpy as np


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.length_base_dice = 0.00001
        self.length_neg_score = 0.00001
        self.length_pos_score = 0.00001

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        batch = probs.shape[0]
        dice, dice_neg, dice_pos, num_neg, num_pos = metric(probs, targets, self.base_threshold)

        self.base_dice_scores.append(dice * batch)
        if num_pos > 0:
            self.dice_pos_scores.append(dice_pos * num_pos)
        if num_neg > 0:
            self.dice_neg_scores.append(dice_neg * num_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

        self.length_base_dice += batch
        self.length_neg_score += num_neg
        self.length_pos_score += num_pos

    def get_metrics(self):
        dice = sum(self.base_dice_scores) / self.length_base_dice
        dice_neg = sum(self.dice_neg_scores) / self.length_neg_score
        dice_pos = sum(self.dice_pos_scores) / self.length_pos_score
        iou = np.nanmean(self.iou_scores)
        metrics = dict(
            dice=dice,
            dice_neg=dice_neg,
            dice_pos=dice_pos,
            iou=iou
        )
        return metrics


def epoch_log(epoch, epoch_loss, meter):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Epoch: %d | Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (epoch,
           epoch_loss, dice, dice_neg, dice_pos, iou))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou
