
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadSegmentationDatasetV4(Dataset):
    """
    PyTorch Dataset for road segmentation training (V4).

    Expects pre-tiled 512x512 image/mask pairs produced by preprocess.py.
    Images are stored as JPEG; masks are stored as grayscale PNG files with
    the same base name as their corresponding image.

    Args:
        image_dir (str): Path to the directory containing image tiles (.jpg).
        mask_dir  (str): Path to the directory containing mask tiles (.png).
        files     (list[str]): List of image filenames to include.
        transform (albumentations.Compose, optional): Augmentation pipeline.
    """

    def __init__(self, image_dir, mask_dir, files, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Masks share the same base name but use the .png extension.
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image as RGB (OpenCV reads BGR by default).
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as single-channel grayscale and binarise to float32.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask has an explicit channel dimension (C, H, W).
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask


def get_transforms(train=False):
    """
    Returns the Albumentations augmentation pipeline for V4.

    Training augmentations include geometric, deformation, photometric,
    and noise transforms. Deformation transforms (GridDistortion,
    ElasticTransform) are particularly important for teaching the model
    to maintain road connectivity under terrain-induced image warps.

    Validation transforms apply only the normalisation required by the
    EfficientNet-B4 ImageNet-pretrained encoder.

    Args:
        train (bool): If True, returns the full training augmentation pipeline.
                      If False, returns the minimal validation pipeline.

    Returns:
        albumentations.Compose: Configured augmentation pipeline.
    """
    if train:
        return A.Compose([
            # --- Geometric / spatial transforms ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Transpose(p=0.5),

            # --- Deformation transforms ---
            # Simulates terrain relief distortions in satellite imagery.
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),

            # --- Colour / lighting transforms ---
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(p=0.2),

            # --- Noise / blur transforms ---
            # Simulates lower-quality or compressed satellite imagery.
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),

            # --- Normalisation (ImageNet statistics for EfficientNet-B4) ---
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
