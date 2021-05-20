from torch import Tensor

from loss_functions import CE_LOSS
from metrics import iou_coef, dice_coef
from model import UNet
import config

import numpy as np
import torch.nn as nn
import torch
import torchvision


# calculate loss and metrics
def calculate_loss(output, target, loss=config.LOSS_FUNCTION):
    if loss == 'cross_entropy_loss':
        return CE_LOSS(output, target)


def calculate_metrics(output, target):
    iou = iou_coef(output, target)
    dice = dice_coef(output, target)

    metrics_dict = {
        "iou_score": iou,
        "dice_score": dice
    }

    return metrics_dict


# load model
def get_model(model_name='unet') -> UNet:
    if model_name == 'unet':
        model = UNet(in_channels=3, out_channels=4)
        return model


def to_rgb(data):
    data = data.astype(int)
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    rgb[data == 1, :] = [255, 0, 0]  # red
    rgb[data == 2, :] = [0, 255, 0]  # green
    rgb[data == 3, :] = [0, 0, 255]  # blue

    return rgb


def get_grid_samples(inputs, targets, outputs):
    inputs = inputs.permute(0, 3, 1, 2)
    targets = [to_rgb(target.numpy()) for target in targets]
    targets = torch.Tensor(targets)

    softmax = nn.Softmax(dim=1)
    predictions = softmax(outputs)
    predictions = torch.argmax(predictions, dim=1)

    predictions = [to_rgb(prediction.numpy()) for prediction in predictions]
    predictions = torch.Tensor(predictions)

    # 4Dminibatch Tensor of shape(Bx C x H x W)
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)
    input_grid = torchvision.utils.make_grid(inputs, nrow=config.BATCH_SIZE)
    target_grid = torchvision.utils.make_grid(targets, nrow=config.BATCH_SIZE)
    output_grid = torchvision.utils.make_grid(predictions, nrow=config.BATCH_SIZE)

    return input_grid.numpy(), target_grid.numpy(), output_grid.numpy()
