
"""
    Meter class for recording data for training and validation dataset:
    To avoid computing data in a whole batch, we update data in each batch with "update" method,
    After completing one epoch, we log the metric for one epoch, at the same time, we delete batch data
    and update epoch data with "epoch_log" method
"""

import os
import pdb
import numpy as np

import torch

class Meter(object):
    """

    """
    def __init__(self, phase, threshold = 0.5):
        self.phase = phase
        self.base_threshold = threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.batch_dice_scores = []
        self.batch_dice_neg_scores = []
        self.batch_dice_pos_scores = []
        self.batch_iou_scores = []

    def update(self, outputs, targets):
        pdb.set_trace()
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = self._cal_metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = self._get_prediction(probs, self.base_threshold)
        iou = self._cal_iou(preds, targets.numpy(), classes=[1])
        self.iou_scores.append(iou)

    def epoch_log(self, epoch, loss, time):
        pdb.set_trace()
        batch_dice_score, batch_dice_neg, batch_dice_pos, batch_iou_score = \
            self._cal_epoch_avg(self.base_dice_scores, self.dice_neg_scores,
                                self.dice_pos_scores, self.iou_scores)

        self.batch_dice_scores.append(batch_dice_score)
        self.batch_dice_neg_scores.append(batch_dice_neg)
        self.batch_dice_pos_scores.append(batch_dice_pos)
        self.batch_iou_scores.append(batch_iou_score)

        del self.base_dice_scores[:]
        del self.dice_neg_scores[:]
        del self.dice_pos_scores[:]
        del self.iou_scores[:]
        print(f"Phase : {self.phase}, Epoch: {epoch}, Time : {time}")
        print(f"Loss: {loss} | dice_pos_score: {batch_dice_pos} | dice_neg_score: {batch_dice_neg} | "
              f"dice_score: {batch_dice_score} |  iou: {batch_iou_score}")

    def _cal_epoch_avg(self, dice_score, dice_neg, dice_pos, iou_score):
        pdb.set_trace()
        batch_dice_score = sum(dice_score) / len(dice_score)
        batch_dice_neg = sum(dice_neg) / len(dice_neg)
        batch_dice_pos = sum(dice_pos) / len(dice_pos)
        iou_score = np.array(iou_score)
        batch_iou_score = np.nanmean(iou_score, axis=0)

        return batch_dice_score, batch_dice_neg, batch_dice_pos, batch_iou_score

    def _cal_metric(self, probs, targets, threshold = 0.5, reductiom = "none"):
        pdb.set_trace()
        batch_size = targets.shape[0]
        probability = probs.view(batch_size, -1)
        truth = targets.view(batch_size, -1)
        assert (probability.shape == truth.shape), "outputs and targets shape don't match"

        p = (probability > threshold).float()
        t = (truth > threshold).float()
        p_sum = torch.sum(p, dim=-1)
        t_sum = torch.sum(t, dim=-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(dim=-1) / ((p + t).sum(-1))
        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

        return dice, dice_neg, dice_pos, num_neg, num_pos

    def _get_prediction(self, probs, threshold):
        pdb.set_trace()
        preds = (probs > threshold).float()
        return preds

    def _cal_iou(self, outputs, labels, classes, only_present = True):
        pdb.set_trace()
        ious = []
        for c in classes:
            label_c = labels == c
            if only_present and np.sum(label_c) == 0:
                ious.append(np.nan)
                continue
            pred_c = outputs == c
            intersection = np.logical_and(pred_c, label_c).sum()
            union = np.logical_or(pred_c, label_c).sum()
            if union != 0:
                ious.append(intersection / union)

        return ious if ious else [1]










