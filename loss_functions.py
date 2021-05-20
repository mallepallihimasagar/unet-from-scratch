import torch
import torch.nn as nn



def CE_LOSS(output, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    ce_loss = cross_entropy_loss(output, target)

    return ce_loss


if __name__ == '__main__':
    a = torch.rand(7, 3, 3, 3)
    b = torch.empty(7, 3, 3, dtype=torch.long).random_(2)

    loss = CE_LOSS(a, b)

    print(loss)
