
"""
Sanchari V4 - Pipeline Validation Tests (test_scripts_v4.py)

Unit tests covering the core V4 pipeline components:
    - Model instantiation and forward pass shape.
    - ComboLoss correctness (perfect prediction = near-zero loss).
    - Dataset loading and tensor shape.
    - Sliding-window inference output shape.

Usage:
    python -m pytest test_scripts_v4.py -v
    python test_scripts_v4.py
"""

import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import rasterio
import torch

from model_v4 import create_model_v4
from train_v4 import ComboLoss
import dataset_v4
from predict_v4 import predict_sliding_window


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def create_dummy_image(path, size=(512, 512), color=(255, 0, 0)):
    """
    Writes a solid-colour BGR image to disk for use in tests.

    Args:
        path  (str):          Output file path.
        size  (tuple[int]):   (height, width) in pixels.
        color (tuple[int]):   BGR colour tuple.
    """
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestV4Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates a temporary directory with a dummy image-mask pair."""
        cls.test_dir  = "test_env_v4"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Dummy image: 1024x1024 to exercise the sliding-window path.
        cls.img_path = os.path.join(cls.test_dir, "test_img.jpg")
        create_dummy_image(cls.img_path, size=(1024, 1024))

        # Dummy mask: grayscale PNG with a small white region.
        cls.mask_path = os.path.join(cls.test_dir, "test_img.png")
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        cv2.imwrite(cls.mask_path, mask)

    @classmethod
    def tearDownClass(cls):
        """Removes the temporary test directory."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    # ------------------------------------------------------------------

    def test_model_creation(self):
        """Verifies that the V4 model instantiates and produces the correct output shape."""
        print("\n[TEST] V4 model creation")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("    Running on CPU (may be slow).")

        try:
            model = create_model_v4().to(device)
            self.assertIsNotNone(model)

            dummy = torch.randn(1, 3, 512, 512).to(device)
            output = model(dummy)
            self.assertEqual(output.shape, (1, 1, 512, 512), "Output shape mismatch.")
            print("    PASS: output shape (1, 1, 512, 512) verified.")
        except Exception as e:
            self.fail(f"Model creation failed: {e}. Ensure 'timm' is installed.")

    # ------------------------------------------------------------------

    def test_combo_loss(self):
        """
        Verifies that ComboLoss returns near-zero loss for a perfect prediction
        (high-confidence logits matching all-ones target).
        """
        print("\n[TEST] ComboLoss")
        criterion = ComboLoss(alpha=0.5, beta=0.5)

        target     = torch.ones((1, 1, 64, 64))
        pred_logits = torch.ones((1, 1, 64, 64)) * 10.0  # High-confidence positive logits.

        loss = criterion(pred_logits, target)
        self.assertLess(loss.item(), 0.1, "Perfect prediction should yield near-zero loss.")
        print(f"    PASS: loss = {loss.item():.6f} (< 0.1).")

    # ------------------------------------------------------------------

    def test_dataset_loading(self):
        """Verifies that the dataset loads the dummy image and returns correct tensor shapes."""
        print("\n[TEST] Dataset loading")
        ds = dataset_v4.RoadSegmentationDatasetV4(
            image_dir=self.test_dir,
            mask_dir=self.test_dir,
            files=["test_img.jpg"],
            transform=dataset_v4.get_transforms_v4(train=False),
        )

        self.assertEqual(len(ds), 1)

        try:
            img, mask = ds[0]
            # The dummy image is 1024x1024; with no resize transform, tensors retain that size.
            self.assertEqual(img.shape,  (3, 1024, 1024), "Image tensor shape mismatch.")
            self.assertEqual(mask.shape, (1, 1024, 1024), "Mask tensor shape mismatch.")
            print("    PASS: tensor shapes (3, 1024, 1024) and (1, 1024, 1024) verified.")
        except Exception as e:
            self.fail(f"Dataset loading failed: {e}")

    # ------------------------------------------------------------------

    def test_sliding_window_inference(self):
        """
        Verifies that the sliding-window function returns a probability map
        matching the spatial dimensions of the input image.

        Uses a lightweight dummy model to avoid loading EfficientNet from disk.
        """
        print("\n[TEST] Sliding-window inference")
        IMG_SIZE = 1024
        device   = torch.device("cpu")

        dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        class DummyModel(torch.nn.Module):
            """Passes through a zero tensor with the correct output shape."""
            def forward(self, x):
                return torch.zeros(x.size(0), 1, x.size(2), x.size(3))

        prob_map = predict_sliding_window(dummy_image, DummyModel(), device)

        self.assertEqual(prob_map.shape, (IMG_SIZE, IMG_SIZE), "Probability map shape mismatch.")
        print(f"    PASS: probability map shape ({IMG_SIZE}, {IMG_SIZE}) verified.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
