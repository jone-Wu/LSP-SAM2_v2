# Copyright (c) SJTU, Z.Wu.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/jone-Wu/LSP-SAM2_v2
import torch.nn.functional as F
import torch
import math
import torch.nn as nn


class DynamicInterpIoULoss(nn.Module): # D-InterpIoU
    def __init__(self, alpha=0.6, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        :param pred: box[N,4](x1,y1,x2,y2)
        :param target: ground truth box[N,4]
        :return: D-interp loss value
        """
        # Calculation of IoU
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        inter = self.intersection(pred, target)
        union = self.union(pred, target)
        iou = (inter + self.eps) / (union + self.eps)
        dynamic_alpha = self.alpha * (1 - iou.detach()).clamp(0, 1)
        # Generate interpolation box[N,4]
        interp_boxes = (dynamic_alpha.unsqueeze(-1) * pred + (1 - dynamic_alpha.unsqueeze(-1)) * target)
        # Calculating IoU of interpolation box(Bint)
        interp_inter = self.intersection(interp_boxes, target)
        interp_union = self.union(interp_boxes, target)
        interp_iou = (interp_inter + self.eps) / (interp_union + self.eps)

        loss = 1 - interp_iou
        return loss[0], iou[0]

    def intersection(self, box1, box2):
        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    def union(self, box1, box2):
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        return area1 + area2 - self.intersection(box1, box2)


class over_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def dice_loss(self, input, target, smooth=1e-5):
        input = torch.sigmoid(input)
        # input = input.flatten()
        # target = target.flatten()
        numerator = 2 * (input * target).sum()
        denominator = input.sum() + target.sum()
        dice_loss = 1 - (numerator + smooth) / (denominator + smooth)
        return dice_loss.sum()

    def sigmoid_focal_loss(
        self,
        input,
        target,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        prob = input.sigmoid()
        p_t = prob * target + (1 - prob) * (1 - target)
        a_t = alpha * target + (1 - alpha) * (1 - target)
        focal_loss = a_t * ce_loss * ((1 - p_t) ** gamma)
        return focal_loss.mean()

    def eiou_loss(self, box1, box2, alpha=1, eps=1e-7):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = torch.pow(inter / (union + eps), alpha)

        # Find the largest enclosing box
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
        rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
        rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
        rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
        cw2 = torch.pow(cw ** 2 + eps, alpha)
        ch2 = torch.pow(ch ** 2 + eps, alpha)
        eiou_loss = 1 - iou + rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2
        return eiou_loss[0], iou  # EIou

    def d_interpiou_loss(self, pred_box, target_box): # box need dtype=torch.float32
        d_interpiou_loss= DynamicInterpIoULoss()
        dinterp_loss, iou = d_interpiou_loss(pred_box, target_box)
        return dinterp_loss, iou




