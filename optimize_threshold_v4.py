
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Import V4 modules
from model_v4 import create_model_v4
from dataset_v4 import RoadSegmentationDatasetV4, get_transforms_v4

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": 'data/processed_v4/train',
    "MODEL_PATH": 'weights/best_model_v4.pth',
    "BATCH_SIZE": 8,
    "VALIDATION_SPLIT": 0.15,
}

def iou_metric_numpy(preds, labels, threshold=0.5):
    """
    Calculates IoU for a batch of predictions using a specific threshold.
    """
    preds_bin = (preds > threshold).astype(np.uint8)
    labels_bin = (labels > 0.5).astype(np.uint8)
    
    intersection = (preds_bin & labels_bin).sum()
    union = (preds_bin | labels_bin).sum()
    
    return (intersection + 1e-6) / (union + 1e-6)

def optimize_threshold(model, loader, device):
    """
    Runs inference on validation set and finds the best threshold.
    """
    model.eval()
    
    # Store all raw probabilities and targets
    all_probs = []
    all_targets = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(device)
            # Get raw logits
            logits = model(data)
            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Use only a subset of pixels to save memory if needed, 
            # but for accuracy best to use all or per-image IoU.
            # Calculating per-image IoU later is better.
            
            # To save memory, we can compute IoU for this batch for ALL thresholds
            # and accumulate stats.
            
            all_probs.append(probs)
            all_targets.append(targets)
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    print(f"Evaluated {len(all_probs)} images.")
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    ious = []
    
    print("Calculating IoUs for different thresholds...")
    for t in thresholds:
        # Calculate mean IoU over the dataset
        # We process in chunks to avoid massive memory usage if array is huge?
        # Actually 1000 images * 512 * 512 is big (~250M pixels).
        # np.uint8 is small.
        
        # Calculate IoU
        # Vectorized batch calculation
        batch_ious = []
        # Loop to save memory
        for i in range(len(all_probs)):
             iou = iou_metric_numpy(all_probs[i], all_targets[i], threshold=t)
             batch_ious.append(iou)
             
        mean_iou = np.mean(batch_ious)
        ious.append(mean_iou)
        print(f"Threshold {t:.2f}: Mean IoU = {mean_iou:.4f}")
        
    best_idx = np.argmax(ious)
    best_threshold = thresholds[best_idx]
    best_iou = ious[best_idx]
    
    print(f"\n--- Results ---")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best IoU: {best_iou:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, ious, marker='o')
    plt.title(f"IoU vs Threshold (V4 Model)\nBest: {best_threshold:.2f} (IoU={best_iou:.4f})")
    plt.xlabel("Threshold")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.savefig("threshold_optimization_v4.png")
    print("Saved plot to threshold_optimization_v4.png")
    
    return best_threshold

def main():
    parser = argparse.ArgumentParser(description="Optimize Threshold V4")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare Data (Validation Set)
    img_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    
    if not os.path.exists(img_dir):
        print(f"Error: Data not found at {img_dir}")
        return

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    train_files, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)
    
    print(f"Validation Samples: {len(val_files)}")

    val_dataset = RoadSegmentationDatasetV4(img_dir, mask_dir, val_files, get_transforms_v4(train=False))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

    # Load Model
    model = create_model_v4().to(device)
    if os.path.exists(CONFIG["MODEL_PATH"]):
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
        print(f"Loaded V4 weights from {CONFIG['MODEL_PATH']}")
    else:
        print(f"Error: Weights not found at {CONFIG['MODEL_PATH']}")
        return

    # Optimize
    optimize_threshold(model, val_loader, device)

if __name__ == "__main__":
    main()
