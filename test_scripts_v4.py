
import unittest
import torch
import numpy as np
import os
import shutil
import cv2
from unittest.mock import MagicMock, patch
import rasterio

# Import Modules to Test
from model_v4 import create_model_v4
from train_v4 import ComboLoss
import dataset_v4
from predict_v4 import predict_sliding_window

# Helper for creating dummy images
def create_dummy_image(path, size=(512, 512), color=(255, 0, 0)):
    # V4 uses 512x512 tiles, input images can be any size, let's use 1024x1024 to test sliding window
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(path, img)

class TestV4Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup temporary environment for file-based tests."""
        cls.test_dir = "test_env_v4"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create dummy image (1024x1024)
        cls.img_path = os.path.join(cls.test_dir, "test_img.jpg")
        create_dummy_image(cls.img_path, size=(1024, 1024))

        # Create binary mask (grayscale)
        # dataset_v4 expects {filename}.png
        cls.mask_path = os.path.join(cls.test_dir, "test_img.png")
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        cv2.imwrite(cls.mask_path, mask)

    @classmethod
    def tearDownClass(cls):
        """Cleanup."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_model_creation(self):
        """Test if the V4 model (EfficientNet-B4 + U-Net++) instantiates correctly."""
        print("\n[TEST] V4 Model Creation")
        if not torch.cuda.is_available():
            print("   [INFO] Testing on CPU (might be slow)")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model = create_model_v4().to(device)
            self.assertIsNotNone(model)
            
            # Test Forward Pass with 512x512 input
            dummy_input = torch.randn(1, 3, 512, 512).to(device)
            output = model(dummy_input)
            self.assertEqual(output.shape, (1, 1, 512, 512), "Output shape mismatch")
            print("   [PASS] Model output shape correct (1, 1, 512, 512)")
        except Exception as e:
            self.fail(f"Model creation failed: {e}. Check if 'timm' is installed.")

    def test_combo_loss(self):
        """Test Combo Loss (Dice + Focal)."""
        print("\n[TEST] Combo Loss")
        criterion = ComboLoss(alpha=0.5, beta=0.5)
        
        # Logits input expected for Focal Loss stability in Combo
        # Case 1: Perfect prediction (Large positive logits for 1, Large negative for 0)
        # Note: BCEWithLogits takes logits.
        
        # Target: All 1s
        target = torch.ones((1, 1, 64, 64))
        # Pred: High confidence logits
        pred_logits = torch.ones((1, 1, 64, 64)) * 10.0
        
        loss = criterion(pred_logits, target)
        self.assertLess(loss.item(), 0.1, "Perfect prediction should have low loss")
        print("   [PASS] Perfect prediction yields low loss")

    def test_dataset_loading(self):
        """Test V4 Dataset class loading and transforms."""
        print("\n[TEST] Dataset Loading")
        ds = dataset_v4.RoadSegmentationDatasetV4(
            image_dir=self.test_dir,
            mask_dir=self.test_dir, 
            files=["test_img.jpg"],
            transform=dataset_v4.get_transforms_v4(train=False)
        )
        
        self.assertEqual(len(ds), 1)
        
        try:
            img, mask = ds[0]
            # Check shapes: V4 transforms resize/crop? No, preprocess usually tiles.
            # But here we load a 1024x1024 image. Albumentations without Resize will keep it.
            # However, preprocess_v4 handles tiling. The Dataset expects TILES.
            # If we pass a large image, it will just load it. 
            # Our dummy image is 1024x1024.
            # ToTensorV2 will make it tensor.
            
            self.assertEqual(img.shape, (3, 1024, 1024), "Image tensor shape (passed raw large image)")
            self.assertEqual(mask.shape, (1, 1024, 1024), "Mask tensor shape")
            
            print("   [PASS] Dataset loaded correctly")
        except Exception as e:
            self.fail(f"Dataset loading failed: {e}")

    def test_sliding_window_inference(self):
        """Test V4 Sliding Window Inference logic."""
        print("\n[TEST] Sliding Window Inference (V4)")
        img_size = 1024
        # Patch size 512
        
        device = torch.device('cpu')
        dummy_large = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Use a small dummy model to avoid loading EfficientNet (slow/heavy)
        # Just need something that outputs (B, 1, 512, 512) given (B, 3, 512, 512)
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.size(0), 1, x.size(2), x.size(3)))
        
        dummy_model = DummyModel()

        prob_map = predict_sliding_window(dummy_large, dummy_model, device)
        
        self.assertEqual(prob_map.shape, (img_size, img_size), "Probability map shape mismatch")
        print("   [PASS] Sliding Window executed successfully")

if __name__ == "__main__":
    unittest.main()
    import timm
    print("Timm Version:", timm.__version__)
    models = timm.list_models('*efficientnet*')
    print(f"Found {len(models)} EfficientNet models.")
    print("First 10 matches:", models[:10])

    # Check specifically for b4
    print("Has 'efficientnet-b4'?", 'efficientnet-b4' in models)
    print("Has 'efficientnet_b4'?", 'efficientnet_b4' in models)
    print("Has 'tf_efficientnet_b4'?", 'tf_efficientnet_b4' in models)
