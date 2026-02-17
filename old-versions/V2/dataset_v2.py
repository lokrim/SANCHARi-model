
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed road segmentation data.
    This is the V2 dataset, including enhanced augmentations.
    """
    def __init__(self, image_dir, mask_dir, image_filenames, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            image_filenames (list): List of image filenames for this dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct paths
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary [0, 1]
        mask = (mask == 255).astype(np.float32)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add a channel dimension to the mask for the loss function
        mask = mask.unsqueeze(0)

        return image, mask

def get_transforms(train=True):
    """
    Returns a composition of Albumentations transforms.
    V2 includes more aggressive augmentations for training.
    """
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # New V2 augmentations for robustness
            A.GridDistortion(p=0.2),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            
            # Normalization and tensor conversion
            A.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet mean
                std=[0.229, 0.224, 0.225],  # ImageNet std
                max_pixel_value=255.0,
            ),
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
