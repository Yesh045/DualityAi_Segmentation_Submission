import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Duality AI Class mapping based on the problem statement
DUALITY_CLASS_IDS = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
ID_TO_INDEX = {class_id: idx for idx, class_id in enumerate(DUALITY_CLASS_IDS)}

class DesertSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (IMREAD_UNCHANGED in case IDs like 7100/10000 are saved in 16-bit)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Map raw Duality IDs to 0-9 for PyTorch CrossEntropy
        mask = np.zeros_like(mask_raw, dtype=np.int64)
        for class_id, class_index in ID_TO_INDEX.items():
            mask[mask_raw == class_id] = class_index

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.to(torch.long)
        return image, mask

def get_training_augmentation(image_size=512):
    return A.Compose([
        A.RandomCrop(width=image_size, height=image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.4), # Small rotation
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augmentation(image_size=512):
    return A.Compose([
        A.CenterCrop(width=image_size, height=image_size, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])