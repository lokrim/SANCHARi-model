
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import V3 modules
from model_v3 import create_model_v3
from dataset_v3 import RoadSegmentationDatasetV3, get_transforms_v3

# Configuration
# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": 'data/processed/train', # Directory containing validation data
    "MODEL_PATH": 'weights/best_model_v3.pth',          # Path to the trained V3 model
    "BATCH_SIZE": 16,                             # Batch size for inference
    "VALIDATION_SPLIT": 0.15                      # Must match training split to get same validation set
}

"""
Threshold Optimization Script for V3 Model.

This script calculates the optimal probability threshold for binary classification
by evaluating the Intersection over Union (IoU) score across a range of thresholds.

Key Features:
1. Loads the trained V3 model.
2. Recreates the exact Validation Set used during training (using same random seed).
3. Runs inference on the validation set with Test Time Augmentation (4-way TTA).
4. Grid searches for the threshold (0.1 to 0.9) that maximizes IoU.
5. Applies the full morphological post-processing pipeline during evaluation to ensure
   the threshold is optimized for the final output quality.
"""

def calculate_iou(preds, labels, threshold):
    """
    Calculates the IoU score for a given threshold, including morphological post-processing.
    
    Args:
        preds (np.array): Array of probability maps (N, H, W).
        labels (np.array): Array of ground truth binary masks (N, H, W).
        threshold (float): Probability threshold to apply.
        
    Returns:
        float: The mean IoU score.
    """
    from skimage.morphology import remove_small_objects, closing, disk
    
    # preds: (N, H, W) probabilities
    # labels: (N, H, W) binary
    
    preds_bin = (preds > threshold)
    
    # Apply morphology (this will be slow but accurate to what predict_v3 does)
    # Since we are doing it on the whole validation set array, it might be memory intensive.
    # It's better to process image by image if N is large, but for standard val set it might fit.
    # Actually, skimage functions work on single images or batches? 
    # remove_small_objects works on boolean array. If 3D (N,H,W), it treats it as 3D object? 
    # YES. We must iterate to behave like 2D prediction.
    
    # Optimization: To avoid 3D connectivity issues, iterate or use a trick.
    # Let's simple iterate for correctness.
    
    intersections = 0.0
    unions = 0.0
    
    for i in range(len(preds_bin)):
        mask = preds_bin[i]
        
        # 1. Remove small noise
        try:
            mask = remove_small_objects(mask, max_size=100)
        except TypeError:
             mask = remove_small_objects(mask, min_size=100)
             
        # 2. Close gaps
        mask = closing(mask, footprint=disk(3))
        
        lbl = labels[i]
        intersections += (mask & lbl).sum()
        unions += (mask | lbl).sum()

    return (intersections + 1e-6) / (unions + 1e-6)

def main():
    """
    Main execution flow.
    1. Prepare Validation Data.
    2. Load Model.
    3. Generate Predictions (with TTA).
    4. Perform Grid Search for Threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Optimizing threshold using device: {device}")

    # 1. Setup Data (Validation Set)
    image_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Use same seed as training to get the same validation set
    _, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)
    
    val_dataset = RoadSegmentationDatasetV3(image_dir, mask_dir, val_files, get_transforms_v3(train=False))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2)
    
    print(f"Evaluating on {len(val_dataset)} validation images.")

    # 2. Load Model
    model = create_model_v3().to(device)
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"Error: {CONFIG['MODEL_PATH']} not found.")
        return
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
    model.eval()

    # 3. Collect Predictions and Targets
    all_preds = []
    all_targets = []

    print("Running inference with TTA...")
    from skimage.morphology import remove_small_objects, closing, disk
    
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data = data.to(device)
            # TTA: Test Time Augmentation
            # 1. Original
            logits = model(data)
            probs = torch.sigmoid(logits)
            
            # 2. Horizontal Flip
            data_h = torch.flip(data, [3])
            logits_h = model(data_h)
            probs_h = torch.flip(torch.sigmoid(logits_h), [3])
            
            # 3. Vertical Flip
            data_v = torch.flip(data, [2])
            logits_v = model(data_v)
            probs_v = torch.flip(torch.sigmoid(logits_v), [2])
            
            # 4. Rotate 90
            data_rot = torch.rot90(data, 1, [2, 3])
            logits_rot = model(data_rot)
            probs_rot = torch.rot90(torch.sigmoid(logits_rot), -1, [2, 3])
            
            # Average
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            
            # To Numpy
            probs_np = probs_avg.cpu().numpy().squeeze(1)
            targets_np = target.cpu().numpy().squeeze(1).astype(bool)

            # Post-processing per image in batch
            # Note: Threshold optimization is tricky with morphology because morphology is binary.
            # However, morphology happens AFTER thresholding.
            # To correctly optimize threshold WITH morphology, we must apply threshold inside the optimization loop.
            # But that is too slow to re-run morphology for every threshold step.
            # Strategy: 
            # 1. We essentially optimizing the "base" threshold.
            
            all_preds.append(probs_np)
            all_targets.append(targets_np)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 4. Grid Search for Best Threshold
    print("Searching for optimal threshold...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_iou = 0
    best_thresh = 0.5

    for t in thresholds:
        iou = calculate_iou(all_preds, all_targets, t)
        print(f"Threshold {t:.2f}: IoU = {iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_thresh = t

    print(f"\n--- Result ---")
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Best IoU:       {best_iou:.4f}")
    print(f"Improvement:    {best_iou - calculate_iou(all_preds, all_targets, 0.5):.4f} over default (0.5)")

if __name__ == "__main__":
    main()
