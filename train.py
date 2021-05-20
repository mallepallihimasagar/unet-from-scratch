import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import config
from utils import calculate_loss, calculate_metrics, get_model, get_grid_samples
from nissl_dataset import NisslDataset


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# load the data
train_data = NisslDataset(root_dir=config.TRAIN_DATA_PATH, transform=True)
test_data = NisslDataset(root_dir=config.TEST_DATA_PATH, transform=False)

train_loader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=config.BATCH_SIZE, shuffle=False)

# get model
model = get_model(model_name=config.MODEL_NAME)
if config.PRE_TRAINED:
    model.load_state_dict(torch.load(config.PRETRAINED_PATH))



# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# intialize wandb
if config.USE_WANDB:
    wandb.init(project=f'{config.WANDB_PROJECT_NAME if config.WANDB_PROJECT_NAME else config.MODEL_NAME}')


# training loop
def train_model(model, train_loader, test_loader, loss_function, calc_metrics):
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

    model.train()
    running_loss = 0
    best_test_loss = 1e+10
    model_weights = model.state_dict()

    for epoch in range(config.NUM_EPOCHS):
        running_loss = 0
        idx = 0
        for idx, data in enumerate(train_loader):
            inputs, target = next(iter(train_loader))  # data
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            loss = loss_function(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / (idx + 1)
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS} - Training Loss = {epoch_loss}')

        metrics = test_model(model, test_loader, loss_function, calc_metrics, scheduler)
        model.train()
        if metrics["loss"] <= best_test_loss:
            print(f'Saving model at epoch :{epoch + 1}')
            model_weights = model.state_dict()
        print(
            f'Epoch {epoch + 1}/{config.NUM_EPOCHS} - Test_loss= {metrics["loss"]}, iou = {metrics["iou_score"]}, dice = {metrics["iou_score"]}')

        if config.USE_WANDB:
            wandb_dict = {
                "training_loss": running_loss / (idx + 1),
                "test_loss": metrics["loss"],
                "iou": metrics["iou_score"],
                "dice": metrics["dice_score"],
                "input": wandb.Image(metrics["input_grid"], caption='Input Batch'),
                "target": wandb.Image(metrics["target_grid"], caption='Target Batch'),
                "output/predictions": wandb.Image(metrics["output_grid"], caption='output/predictions Batch')

            }
            print("logging to wandb")
            wandb.log(wandb_dict)

    print(f'saving best model weights to {config.MODEL_SAVE_PATH}')
    model.load_state_dict(model_weights)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print('Training completed')


def test_model(model, test_loader, loss_function, calc_metrics, scheduler):
    model.eval()
    running_loss = 0
    iou = 0
    dice = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            loss = loss_function(output, target)
            metrics = calc_metrics(output, target)

            running_loss += loss.item()
            iou += metrics["iou_score"]
            dice += metrics["dice_score"]
        scheduler.step(running_loss / (idx + 1))
        input_grid, target_grid, output_grid = get_grid_samples(
            inputs.cpu(),
            target.cpu(),
            output.cpu()
        )
        eval_metrics = {
            "loss": running_loss / (idx + 1),
            "iou_score": iou / (idx + 1),
            "dice_score": dice / (idx + 1),
            "input_grid": input_grid,
            "target_grid": target_grid,
            "output_grid": output_grid
        }
        return eval_metrics


if __name__ == "__main__":
    # loss_function = partial(calculate_loss, config.LOSS_FUNCTION)
    model = get_model(config.MODEL_NAME)
    if device != 'cpu':
        model = model.to(device)

    train_model(model, train_loader, test_loader, calculate_loss, calculate_metrics)
