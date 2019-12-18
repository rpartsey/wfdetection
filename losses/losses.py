import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLogDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def __call__(self, input, target):
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return - ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).log()


class ImageBinaryLogDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def __call__(self, input, target):
        input = torch.sigmoid(input)
        batch = input.shape[0]
        iflat = input.view(batch, -1)
        tflat = target.view(batch, -1)
        intersection = (iflat * tflat).sum(-1)
        return - ((2.0 * intersection + self.smooth) / (iflat.sum(-1) + tflat.sum(-1) + self.smooth)).log().mean()


class OfficialBinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(OfficialBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.float()
        pos = (targets > 0.5).float()
        neg = (targets < 0.5).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        F_loss = self.alpha * pos * F_loss + (1 - self.alpha) * neg * F_loss
        return torch.mean(F_loss)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class OnlyPositiveBinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.sum() / ((target>0.5).float().sum() + 1.0)


class BinaryFocalDice(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0, smooth=1.0, divider=1.0):
        super().__init__()
        self.alpha = alpha
        self.focal = BinaryFocalLoss(gamma)
        self.log_dice_loss = BinaryLogDice(smooth=smooth)
        self.divider = divider
        print("Using divider", self.divider)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) + self.log_dice_loss(input, target)
        return loss / self.divider


class BinaryOnlyPositiveFocalDice(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smooth=1.0, divider=1.0):
        super().__init__()
        self.alpha = alpha
        self.focal = OnlyPositiveBinaryFocalLoss(gamma)
        self.log_dice_loss = BinaryLogDice(smooth=smooth)
        self.divider = divider
        print("Using divider", self.divider)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) + self.log_dice_loss(input, target)
        return loss / self.divider

class BinaryCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()

class BinaryBatchWeightedCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if 0:
            loss = loss.mean()

        if 1:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

        return loss


class WeightedLogDice(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1.0
        self.gamma = gamma
        sum = self.beta + self.alpha
        if sum != 1:
            self.beta = self.beta / sum
            self.alpha = self.alpha / sum

    # @staticmethod
    def forward(self, pred, target):
        target = target.view(-1).float()
        pred = pred.sigmoid().view(-1)
        # _, input_label = input.max(1)
        true_pos = (target * pred).sum()
        false_neg = (target * (1 - pred)).sum()
        false_pos = ((1 - target) * pred).sum()

        loss = (2 * true_pos + self.smooth) / (2 * true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)

        return - loss.log()
