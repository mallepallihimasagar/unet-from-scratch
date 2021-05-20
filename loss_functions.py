import torch
import torch.nn as nn



def CE_LOSS(output, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    ce_loss = cross_entropy_loss(output, target)

    return ce_loss


if __name__ == '__main__':
    a = torch.rand(3, 4, 512, 512)
    b = torch.empty(3, 512, 512, dtype=torch.long).random_(4)

    loss = CE_LOSS(a, b)

    print(loss)
