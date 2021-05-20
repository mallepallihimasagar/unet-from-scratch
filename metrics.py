import torch
import torch.nn as nn


def iou_coef(y_pred, y_true, smooth=1e-4):
    if len(y_true.shape) == 3:
        y_true = y_true.unsqueeze(1)
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[2, 3, 1])
    union = torch.sum(y_true, [2, 3, 1]) + torch.sum(y_pred, [2, 3, 1]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou


def dice_coef(y_pred, y_true, smooth=1e-4):
    if len(y_true.shape) == 3:
        y_true = y_true.unsqueeze(1)
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[2, 3, 1])
    union = torch.sum(y_true, [2, 3, 1]) + torch.sum(y_pred, [2, 3, 1])
    dice = torch.mean((2*intersection + smooth) / (union + smooth), dim=0)
    return dice


if __name__ == '__main__':
    a = torch.ones(2, 1, 10, 10)
    b = torch.zeros(2, 10, 10)

    iou = dice_coef(a,a)
    print(iou)
