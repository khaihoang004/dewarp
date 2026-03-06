import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import random
from torch.utils.data import Dataset

from .utils import load_image, load_mat, crop_tight
from .transforms import get_appearance_transform

class DIWDataset(Dataset):
    """ Evaluation dataset for DIW data. """
    def __init__(self, paths, img_size=(356, 244)):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        # Load image and backward map
        img = load_image(image_path)
        # Load checkerboard map
        mask_path = image_path.replace('/img/', '/seg/')
        cb = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cb = (cb > 0).astype(np.uint8)[..., None]
        img = img * cb  # Apply mask to image
        # Apply transformations
        img = torch.from_numpy(img).permute(2, 0, 1).float()   # (C, H, W)
        # Resize to desired image size
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=True).squeeze(0)
        return img / 255.0

if __name__ == "__main__":
    from torch.utils.data import DataLoader
        # Load DIW dataset paths
    diw_dir = "/kaggle/input/datasets/minhducnguyen9705/document-in-wild-dataset/5k/img"
    test_paths = glob.glob(os.path.join(diw_dir, '*.*'))

    # Create datasets
    batch_size = 8
    # transform = v2.Compose([v2.ToTensor()])
    test_dataset = DIWDataset(test_paths)
    print(f'Test set size: {len(test_dataset)}')

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)