
"""
Sanchari V4 - Threshold Optimisation Script (optimize_threshold_v4.py)

Evaluates the trained V4 model on the validation split across a range of
binarisation thresholds and plots mean IoU vs. threshold to identify the
optimal operating point.

Output:
    threshold_optimization_v4.png -- IoU vs threshold curve.

Usage:
    python optimize_threshold_v4.py
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from model_v4 import create_model_v4
from dataset_v4 import RoadSegmentationDatasetV4, get_transforms_v4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed_v4/train",
    "MODEL_PATH":         "weights/best_model_v4.pth",
    "BATCH_SIZE":         8,
    "VALIDATION_SPLIT":   0.15,
}


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def iou_metric_numpy(preds, labels, threshold=0.5):
    """
    Computes IoU for a single image using a given binarisation threshold.

    Args:
        preds     (np.ndarray): Float32 probability array (C, H, W) or (H, W).
        labels    (np.ndarray): Ground-truth binary array of the same shape.
        threshold (float):      Binarisation threshold.

    Returns:
        float: Scalar IoU value with Laplace smoothing.
    """
    preds_bin  = (preds  > threshold).astype(np.uint8)
    labels_bin = (labels > 0.5).astype(np.uint8)

    intersection = (preds_bin & labels_bin).sum()
    union        = (preds_bin | labels_bin).sum()

    return float(intersection + 1e-6) / float(union + 1e-6)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def optimize_threshold(model, loader, device):
    """
    Collects per-image predictions on the validation set and sweeps thresholds.

    Runs a single inference pass to collect all raw probabilities, then
    evaluates mean IoU at each threshold from 0.10 to 0.90 (step 0.05).
    Saves a curve plot and prints the best threshold found.

    Args:
        model  (nn.Module):    Trained model in eval mode.
        loader (DataLoader):   Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: Best threshold value (highest mean IoU).
    """
    model.eval()
    all_probs   = []
    all_targets = []

    print("Running inference on validation set ...")
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data    = data.to(device)
            logits  = model(data)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    print(f"Collected probabilities for {len(all_probs)} images.")

    thresholds = np.arange(0.10, 0.95, 0.05)
    mean_ious  = []

    print("Sweeping thresholds ...")
    for t in thresholds:
        per_image_ious = [
            iou_metric_numpy(all_probs[i], all_targets[i], threshold=t)
            for i in range(len(all_probs))
        ]
        mean_iou = np.mean(per_image_ious)
        mean_ious.append(mean_iou)
        print(f"  Threshold {t:.2f}: Mean IoU = {mean_iou:.4f}")

    best_idx       = int(np.argmax(mean_ious))
    best_threshold = thresholds[best_idx]
    best_iou       = mean_ious[best_idx]

    print(f"\nBest threshold : {best_threshold:.2f}")
    print(f"Best mean IoU  : {best_iou:.4f}")

    # Save plot.
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mean_ious, marker="o")
    plt.title(f"IoU vs Threshold (V4)\nBest: {best_threshold:.2f}  (IoU = {best_iou:.4f})")
    plt.xlabel("Threshold")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.savefig("threshold_optimization_v4.png")
    print("Saved plot to threshold_optimization_v4.png.")

    return best_threshold


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimise binarisation threshold for V4 model.")
    parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_dir  = os.path.join(CONFIG["PROCESSED_DATA_DIR"], "images")
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Data not found at {img_dir}. Run preprocess_v4.py first.")
        return

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    _, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)
    print(f"Validation samples: {len(val_files)}")

    val_dataset = RoadSegmentationDatasetV4(img_dir, mask_dir, val_files, get_transforms_v4(train=False))
    val_loader  = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

    model = create_model_v4().to(device)
    if os.path.exists(CONFIG["MODEL_PATH"]):
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
        print(f"V4 weights loaded from {CONFIG['MODEL_PATH']}.")
    else:
        print(f"Error: Weights not found at {CONFIG['MODEL_PATH']}.")
        return

    optimize_threshold(model, val_loader, device)


if __name__ == "__main__":
    main()
