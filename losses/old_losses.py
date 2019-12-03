import os

import torch
from torch.autograd import Function, Variable

from torch.nn import functional as F
import numpy as np
import torch.nn as nn


try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse


DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")


def bce_weights(logits, true):
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5*36.548575]).to(DEVICE))
    return loss(logits, true.float())


def cross_entropy_weights(logits, true):
    B, _, H, W = true.size()
    return F.cross_entropy(logits, true.view(B, H, W), weight=torch.tensor([0.506935, 36.548575]).to(DEVICE))

class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


def cross_entropy_dice(logits, true):
    B, _, H, W = true.size()
    return F.cross_entropy(logits, true.view(B, H, W), weight=torch.tensor([0.506935, 36.548575]).to(DEVICE)) * 0.7 + main_dice_loss(logits, true) * 0.3


def torch_dice_loss_metric(y_pred, y_true, smooth = 1.0, do_sigmoid_or_softmax=True):
    y_true = y_true.float()
    y_pred = y_pred[:, 1:, :, :]
    if do_sigmoid_or_softmax:
        y_pred = torch.sigmoid(y_pred)
    assert y_pred.size() == y_true.size()
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (
        y_pred.sum() + y_true.sum() + smooth
    )
    return dsc


def main_dice_loss(logits, true, eps=1e-7, do_sigmoid_or_softmax=True):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.

    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
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
    dims = (0,)
    # dims = []
    # B, C, W, H
    dims += tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean(dim=1)
    return - (dice_loss - 1) #.mean()


def lovasz(input, target):
    lovash = lovasz_softmax(input, target, ignore=None)
    return lovash


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.2, size_average=True, reduce=True, do_sigmoid_or_softmax=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.do_sigmoid_or_softmax = do_sigmoid_or_softmax
        self.reduce = reduce

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        if self.do_sigmoid_or_softmax:
            logpt = F.log_softmax(input)
        else:
            logpt = input.log()
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data).to(input.get_device())
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduce:
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
        return loss


def focal_loss(logits, masks, reduce=True, do_sigmoid_or_softmax=True):
    focal_loss = FocalLoss(do_sigmoid_or_softmax=do_sigmoid_or_softmax, reduce=reduce).forward(logits, masks)
    return focal_loss


def tversky_loss(logits, masks, alpha=0.7, beta=0.3, eps=1e-7, do_sigmoid_or_softmax=True):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[masks.squeeze(1)]
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
        true_1_hot = torch.eye(num_classes)[masks.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        if do_sigmoid_or_softmax:
            probas = F.softmax(logits, dim=1)
        else:
            probas = logits
    true_1_hot = true_1_hot.type(logits.type()).to(logits.get_device())
    dims = (0,) + tuple(range(2, masks.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


def focal_tversky(logits, masks, do_sigmoid_or_softmax=True):
    return 0.5 * tversky_loss(logits, masks, do_sigmoid_or_softmax=do_sigmoid_or_softmax) + 0.5 * focal_loss(logits, masks, do_sigmoid_or_softmax=do_sigmoid_or_softmax)


def focal_lovaszh(logits, masks):
    return 0.5 * focal_loss(logits, masks) + 0.5 * lovasz(logits, masks)

def focal_dice(logits, masks, do_sigmoid_or_softmax=True):
    loss = 10 * focal_loss(logits,masks, reduce=True, do_sigmoid_or_softmax=do_sigmoid_or_softmax) - torch_dice_loss_metric(logits, masks, do_sigmoid_or_softmax=do_sigmoid_or_softmax).log()
    return loss.mean()


#==========================================================HELPERS====================================================================


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


def make_one_hot(labels, device, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().to(device)
    target = one_hot.scatter(1, labels.data, 1)

    target = torch.tensor(target)

    return target

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels




if __name__ == '__main__':
    a = torch.tensor([
            [[1,0],[1,0]],
            [[1,0],[1,0]]
    ]
        ).float()
    b = torch.tensor([
            [[1,0],[1,0]]
    ],
        ).long()
    print(a.shape, b.shape)
    res = main_dice_metric(a, b)
