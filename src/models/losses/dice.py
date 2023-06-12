import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, input, target):
        b = target.size(0)
        smooth = 1e-5
        input_flat = input.view(b, -1)
        target_flat = target.view(b, -1)

        intersection = input_flat * target_flat
        loss = (2*intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = loss.mean()
        loss = 1 - loss
        return loss


class SoftDiceLossWithLogits(SoftDiceLoss):
    def __init__(self):
        super(SoftDiceLossWithLogits, self).__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        loss = super(SoftDiceLossWithLogits, self).forward(input, target)
        return loss
