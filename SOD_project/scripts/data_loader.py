import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

IMG_SIZE = 128

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img / 255.0

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = mask / 255.0
    return mask

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        mask_name = img_name.replace(".jpg", ".png")
        image = preprocess_image(os.path.join(self.img_dir, img_name))
        mask = preprocess_mask(os.path.join(self.mask_dir, mask_name))

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask