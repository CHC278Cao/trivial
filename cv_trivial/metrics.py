
"""
    Loss function for models: before applying loss function,
    use y_preds = out.cpu().detach().numpy() to transfer model output to numpy
    y_preds: type: numpy.ndarray
    targets: type: numpy.ndarray
"""

import math
import torch
import torch.nn.functional as F


def _sharpen(possibility, t=0.5):
    if t != 0:
        return possibility ** t
    else:
        return possibility


def DiceLoss(y_preds, targets, smooth=1):
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    intersection = (y_preds * targets).sum()
    dice = (2. * intersection + smooth) / (y_preds.sum() + targets.sum() + smooth)

    return 1 - dice


def DiceBCELoss(y_preds, targets, smooth = 1):
    batch_size = targets.shape[0]
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    intersection = (y_preds * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (y_preds.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(y_preds, targets.float(), reduction="mean")
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def IoULoss(y_preds, targets, smooth = 1):
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    intersection = (y_preds * targets).sum()
    total = (y_preds + targets).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def FocalLoss(y_preds, targets, alpha = 0.8, gamma = 2, smooth = 1):
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    BCE = F.binary_cross_entropy(y_preds, targets) / y_preds.shape[0]
    BCE_EXP = torch.exp(-BCE)
    FocalLoss = alpha * (1 - BCE_EXP)**gamma * BCE

    return FocalLoss


def TverskyLoss(y_preds, targets, smooth = 1, alpha = 0.5, beta = 0.5):
    """
        Calculate the loss to optimize segmentation on imbalanced dataset, alpha penalizes FP,
        beta penalizes FN, the higher alpha and beta are, the more they penalize
    :param y_preds:     type: np.ndarray, predicted output
    :param targets:     type: np.ndarray, targets
    :param smooth:      type: float, smoothing weight
    :param alpha:       type: float, penalty for FP
    :param beta:        type: float, penalty for FN
    :return:
        TverskyLoss
    """
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    TP = (y_preds * targets).sum()
    FP = ((1 - targets) * y_preds).sum()
    FN = (targets * (1 - y_preds)).sum()
    TverskyLoss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return TverskyLoss


def FocalTverskyLoss(y_preds, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    TP = (y_preds * targets).sum()
    FP = ((1 - targets) * y_preds).sum()
    FN = (targets * (1 - y_preds)).sum()
    loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTverskyLoss = (1 - loss)**gamma

    return FocalTverskyLoss


def ComboLoss(y_preds, targets, smooth=1, alpha=0.5, ce_ratio = 0.5):
    """

    :param y_preds:
    :param targets:
    :param smooth:
    :param alpha:
    :param ce_ratio:
    :return:
    """
    # y_preds = y_preds.view(-1)
    # targets = targets.view(-1)

    intersection = (y_preds * targets).sum()
    dice = (2. * intersection + smooth) / (y_preds.sum() + targets.sum() + smooth)
    y_preds = torch.clamp(y_preds, math.e, 1-math.e)
    out = - (alpha * ((targets * torch.log(y_preds)) + (1 - alpha) * (1 - targets) * torch.log(1 - y_preds)))
    weighted_ce = out.mean(-1)
    comboLoss = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return comboLoss

