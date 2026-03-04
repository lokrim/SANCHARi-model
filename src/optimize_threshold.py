
import sys
import os

# Ensure src/ is on the path so sibling modules (model, dataset) can be imported
# regardless of whether this script is run from the repo root or from src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Import V1 modules
from model import UNet
from dataset import RoadSegmentationDataset, get_transforms

# --- Configuration ---
# Paths are anchored to the repo root via __file__, so this script works
# whether run as `python src/optimize_threshold.py` or `cd src && python optimize_threshold.py`.
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
CONFIG = {
    "PROCESSED_DATA_DIR": os.path.join(_REPO_ROOT, 'data', 'processed', 'train'),
    "MODEL_PATH": os.path.join(_REPO_ROOT, 'weights', 'best_model_v1.pth'),
    "BATCH_SIZE": 32,
    "VALIDATION_SPLIT": 0.15,
}

def iou_metric(preds, labels, threshold):
    """Calculates IoU for a given threshold."""
    preds = torch.sigmoid(preds) > threshold
    preds, labels = preds.byte(), labels.byte()
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

def find_optimal_threshold(model, loader, device):
    """
    Finds the optimal prediction threshold on the validation set.
    """
    model.eval()
    thresholds = np.arange(0.2, 0.8, 0.02) # Test a range of thresholds
    best_iou = 0
    best_threshold = 0

    all_preds = []
    all_targets = []

    print("Gathering predictions from validation set...")
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Predicting"):
            preds = model(data.to(device))
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    print("\nSearching for optimal threshold...")
    for threshold in tqdm(thresholds, desc="Thresholding"):
        iou_scores = [iou_metric(p, t, threshold) for p, t in zip(all_preds, all_targets)]
        current_iou = np.mean(iou_scores)

        if current_iou > best_iou:
            best_iou = current_iou
            best_threshold = threshold
            print(f"  New best IoU: {best_iou:.4f} at threshold: {best_threshold:.2f}")

    return best_threshold, best_iou

def main():
    """
    Main function to load the initial model and find the best threshold.
    """
    print("Starting threshold optimization for the initial model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"Error: Model file not found at {CONFIG['MODEL_PATH']}. Please run train.py first.")
        return
    # Instantiate the original U-Net model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
    print(f"Model loaded from {CONFIG['MODEL_PATH']}")

    # --- Validation Dataset and Dataloader ---
    image_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    
    # The original dataset class loads all files, so we split the filenames first
    all_files = sorted(os.listdir(image_dir))
    _, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)

    # Create the dataset instance
    val_dataset = RoadSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=get_transforms(train=False))
    # Manually assign the correct validation file list to the dataset instance
    val_dataset.images = val_files

    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)
    print(f"Using {len(val_dataset)} validation samples to find the best threshold.")

    # --- Find Threshold ---
    optimal_threshold, max_iou = find_optimal_threshold(model, val_loader, device)

    print("\n--- Optimization Complete ---")
    print(f"Optimal Threshold for initial model: {optimal_threshold:.2f}")
    print(f"Maximum IoU on Validation Set: {max_iou:.4f}")

if __name__ == '__main__':
    main()
