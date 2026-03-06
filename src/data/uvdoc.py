import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .utils import load_image, load_mat, crop_tight
from .transforms import get_appearance_transform

class UVDoc(Dataset):
    """ Custom Dataset for loading UVDoc data with appearance augmentations. """
    def __init__(self, root_dir, img_size=(356, 244), bm_size=(45, 31), appearance_augmentation=[]):
        self.root_dir = root_dir
        self.img_size = img_size
        self.bm_size = bm_size
        self.appearance_transform = get_appearance_transform(appearance_augmentation)
        
        # Precompute path mappings (very important)
        self.samples = []
        for path in glob.glob(os.path.join(root_dir, "metadata_sample/*.json")):
            with open(path, "r") as f:
                metadata = json.load(f)
            sample_name, sample_id = metadata["geom_name"], metadata["sample_id"]
            img_path = os.path.join(self.root_dir, "img", f"{sample_id}.png")
            grid2D_path = os.path.join(self.root_dir, "grid2d", f"{sample_name}.mat")
            seg_path = os.path.join(self.root_dir, "seg", f"{sample_name}.mat")
            self.samples.append((img_path, grid2D_path, seg_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bm_path, cb_path = self.samples[idx]
        # Load image and backward map
        img = load_image(img_path)
        cb = load_mat(cb_path, key="seg")  # Load checkerboard mask
        cb = np.expand_dims(cb, axis=-1)  # Add channel dimension if needed
        img = img * cb  # Apply mask to image
        bm = load_mat(bm_path, key="grid2d")  # Load backward map
        
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

# Initialize dataset
if __name__ == "__main__":
    uvdoc_dataset = UVDoc(root_dir="/kaggle/input/datasets/kahazai/uvdoc-dataset/UVDoc_final/UVDoc_final", img_size=(356, 244), bm_size=(45, 31))
    print(f'Train uvdoc dataset size: {len(uvdoc_dataset)}')
    