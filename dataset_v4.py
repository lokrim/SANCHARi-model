
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadSegmentationDatasetV4(Dataset):
    """
    PyTorch Dataset for Road Segmentation (V4).
    Supports 512x512 inputs and robust augmentations.
    """
    def __init__(self, image_dir, mask_dir, files, transform=None):
        """
        Args:
            image_dir (str): Path to images.
            mask_dir (str): Path to masks.
            files (list):/List of filenames.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Construct paths
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Mask is .png, image is .jpg (based on preprocess_v4)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize (Float32 for Loss)
        mask = (mask > 127).astype(np.float32)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Ensure channel dim
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask

def get_transforms_v4(train=False):
    """
    V4 Augmentation Pipeline.
    Includes 'Hard' augmentations (GridDistortion, ElasticTransform) 
    to prevent overfitting on larger 512x512 patches.
    """
    if train:
        return A.Compose([
            # 1. Spatial / Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Transpose(p=0.5),
            
            # 2. Deformations (Crucial for V4)
            # Helps model learn road connectivity despite warps
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3), # Removed alpha_affine (deprecated)
            
            # 3. Color / Lighting
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.RandomGamma(p=0.2),
            
            # 4. Noise/Blur (Simulate lower quality satellite imagery)
            # var_limit was deprecated/renamed in some versions, using defaults for safety or specific kwargs
            # If var_limit is warning, it might expect a single float or tuple. 
            # Let's use simple GaussNoise without explicit var_limit if it complains, or check docs.
            # Actually, current albumentations uses var_limit. Maybe value format?
            # Let's try standard range. If warning persists, it might be a version quirk.
            # We'll stick to default var_limit by not passing it, defaulting to (10.0, 50.0) usually.
            A.GaussNoise(p=0.2), 
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            
            # 5. Normalization specific to EfficientNet (ImageNet stats)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
