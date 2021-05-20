import torch
from torch.utils.data import Dataset
import albumentations as A

import os
import cv2
from skimage import io
import numpy as np

import config


class NisslDataset(Dataset):
    def __init__(self, root_dir='Nissl_Dataset', transform=True):
        self.root_dir = root_dir
        self.transform = transform
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        path = os.listdir(self.root_dir)[index]
        final_path = os.path.join(self.root_dir, path)

        image = io.imread(f'{final_path}/{path}.png')  # [500,500,3] -> [W,H,C]
        mask1 = io.imread(f'{final_path}/{path}_mask1.png')  # [500,500] -> [W,H]
        mask2 = io.imread(f'{final_path}/{path}_mask2.png')  # [500,500] -> [W,H]
        mask3 = io.imread(f'{final_path}/{path}_mask3.png')  # [500,500] -> [W,H]

        assert mask1.shape == mask2.shape == mask3.shape, f'mask shapes mismatch {mask1.shape},{mask2.shape},{mask3.shape} '
        mask = np.zeros((mask1.shape[0], mask1.shape[1]), dtype=np.uint8)

        mask[mask1 == 255] = 1
        mask[mask2 == 255] = 2
        mask[mask3 == 255] = 3

        image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
        mask = cv2.resize(mask, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.from_numpy(image)  # [W, H, C]
        image = image.permute(2, 0, 1)  # [C, W, H]

        mask = torch.from_numpy(mask)  # [W, H, C]
        #mask = mask.permute(2, 0, 1)  # [C, W, H]

        return image, mask


if __name__ == '__main__':
    data = NisslDataset(root_dir='Nissl_Dataset/train')
    image, mask = data.__getitem__(13)

    print(image.shape)
    print(mask.shape)
