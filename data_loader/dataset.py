import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Custom dataset for image processing"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_names = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        blurred = Image.open(os.path.join(self.data_dir, image_name)).convert('RGB')
        h = blurred.size[1]
        w = blurred.size[0]
        new_h = h - h % 4 + 4 if h % 4 != 0 else h
        new_w = w - w % 4 + 4 if w % 4 != 0 else w
        blurred = transforms.Resize([new_h, new_w], Image.BICUBIC)(blurred)

        if self.transform:
            blurred = self.transform(blurred)

        return {'blurred': blurred, 'image_name': image_name}
