
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadSegmentationDatasetV3(Dataset):
    """
    PyTorch Dataset for Road Segmentation (V3).
    
    Handles loading images and binary masks from the filesystem, applying
    transformations, and preparing tensors for the model.
    """
    def __init__(self, image_dir, mask_dir, files, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing input images (jpg).
            mask_dir (str): Path to the directory containing binary masks (png/jpg).
            files (list): List of filenames (basenames) to include in this dataset 
                          (allows for external train/val splitting logic).
            transform (albumentations.Compose): Augmentation/Preprocessing pipeline.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads and processes one sample (Image + Mask).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct paths
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Construct mask path (Handles different extensions if needed, assumes .png vs .jpg match)
        # DeepGlobe masks often have same basename but might be .png while images are .jpg
        # Here we assume _mask convention or direct name match.
        # Based on preprocess_v3, we saved them with same names.
        # Fix: Preprocess saves masks as .png. Images as .jpg. Base name is otherwise same.
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask (0 or 255 -> 0 or 1)
        # We use a float32 mask for BCE/Dice loss usually
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Ensure mask has channel dimension (H, W) -> (1, H, W) for PyTorch
            # Albumentations doesn't add channel dim to 2D masks by default unless specified
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask

# --- Transformations ---
def get_transforms_v3(train=False):
    """
    Returns the Albumentations transformation pipeline for V3.
    
    Args:
        train (bool): If True, returns a pipeline with heavy data augmentation 
                      (flips, distortions, brightness/contrast) to prevent overfitting.
                      If False, returns only the necessary preprocessing (normalization)
                      for validation/inference.
    
    Returns:
        albumentations.Compose: The composition of transforms.
    """
    if train:
        return A.Compose([
            # Geometric Augmentations (Spatial)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Transpose(p=0.5), # Randomly swap spatial axes
            
            # Deformations (great for winding roads)
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.5),
            
            # Color/Intensity Augmentations (Robustness to lighting)
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            
            # Preprocessing
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Validation/test transforms only include normalization
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    return transform
