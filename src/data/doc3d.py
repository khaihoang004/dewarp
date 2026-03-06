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

class Doc3D(Dataset):
    """ Custom Dataset for loading Doc3D data with appearance augmentations. """
    def __init__(self, paths, img_size=(356, 244), bm_size=(45, 31), appearance_augmentation=[]):
        self.paths = paths
        self.img_size = img_size
        self.bm_size = bm_size
        self.appearance_transform = get_appearance_transform(appearance_augmentation)
        
        # Precompute path mappings (very important)
        self.samples = []
        for p in paths:
            # bm_path = p.replace('img', 'bm').split('.')[-2] + '.mat'
            tags = p.split('/')
            bm_path = os.path.join("/kaggle/input/datasets/kahazai/uvdoc-dataset/Doc3D_grid/Doc3D_grid/grid2D", tags[-2], tags[-1].split('.')[-2] + '.mat')
            cb_path = p.replace('img', 'recon')[:-8] + 'chess480001.png'
            self.samples.append((p, bm_path, cb_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bm_path, cb_path = self.samples[idx]
        # Load image and backward map
        img = load_image(img_path)
        cb = cv2.imread(cb_path, cv2.IMREAD_GRAYSCALE)
        cb = (cb > 0).astype(np.uint8)[..., None]
        img = img * cb  # Apply mask to image
        bm = load_mat(bm_path, 'grid2D')
        
        # Apply appearance augmentations (expects HWC format with float32 in [0, 1])
        if self.appearance_transform is not None:
            augmented = self.appearance_transform(image=img.astype(np.uint8))
            img = augmented['image']
        
        # Apply tight crop based on backward map
        img, bm = crop_tight(img, bm)

        # Apply transformations
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)   # (C, H, W)
        bm = torch.from_numpy(bm).permute(2, 0, 1).float().unsqueeze(0)   # (2, H, W)
        # Apply tight crop based on backward map
        # Resize to desired image size
        img = F.interpolate(img, size=self.img_size, mode='bilinear', align_corners=True).squeeze(0)
        bm = F.interpolate(bm, size=self.bm_size, mode='bilinear', align_corners=True).squeeze(0)
        return img, bm

if __name__ == "__main__":
    ### Load dataset paths
    bad_paths = ["/kaggle/input/datasets/oneplusoneisthree/doc3d-subset-v3/img/21/556_7-ny_Page_183-cvM0001.png", "/kaggle/input/datasets/oneplusoneisthree/doc3d-subset-v3/img/21/2_431_3-cp_Page_0802-Pum0001.png"]
    warped_paths = glob.glob(os.path.join( '/kaggle/input/datasets/alreadytaken23/doc3d-subset/img', '**/*.*'))
    warped_paths += glob.glob(os.path.join( '/kaggle/input/datasets/anguyen023/doc3d-subset-v2/img', '**/*.*'))
    warped_paths += glob.glob(os.path.join( '/kaggle/input/datasets/oneplusoneisthree/doc3d-subset-v3/img', '**/*.*'))
    warped_paths = [p for p in warped_paths if p not in bad_paths]
    print(f'Total images found: {len(warped_paths)}')

    # Split train and validation sets
    ratio = 0.95
    random.shuffle(warped_paths)
    split_idx = int(len(warped_paths) * ratio)
    train_paths = warped_paths[:split_idx]
    val_paths = warped_paths[split_idx:]


    # appearance_augs = ['visual', 'color',] # 'noise', 'shadow', 'blur']
    appearance_augs = [] # Use None for now
    doc3d_dataset = Doc3D(train_paths)
    val_dataset = Doc3D(val_paths)

    print(f'Train doc3d dataset size: {len(doc3d_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')