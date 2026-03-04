
"""
Sanchari V4 - Training Script (train.py)

Main training loop for the U-Net++ / EfficientNet-B4 road segmentation model.

Training phases:
    1. Main training (50 epochs) with Combo Loss (Dice + Focal).
    2. Optional Hard Negative Mining phase (--hard-mining flag).

Usage:
    python train.py
    python train.py --resume
    python train.py --hard-mining
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import numpy as np

from model import create_model
from dataset import RoadSegmentationDatasetV4, get_transforms


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "PROCESSED_DATA_DIR": os.path.join(BASE_DIR, "data/processed/train"),
    "MODEL_SAVE_PATH":    os.path.join(BASE_DIR, "weights/best_model_v4.pth"),
    "CHECKPOINT_PATH":    os.path.join(BASE_DIR, "checkpoint.pth"),
    "LOG_FILE":           os.path.join(BASE_DIR, "training_log.csv"),

    "LEARNING_RATE":           5e-4,
    "WEIGHT_DECAY":            1e-4,
    "BATCH_SIZE":              8,
    "GRAD_ACCUMULATION_STEPS": 2,    # Effective batch size: 8 x 2 = 16.
    "NUM_EPOCHS":              50,
    "SCHEDULER_T_MAX":         50,
    "VALIDATION_SPLIT":        0.15,
}


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class ComboLoss(nn.Module):
    """
    Weighted combination of Dice Loss and Focal Loss.

    Dice Loss optimises the pixel-level overlap between prediction and ground
    truth, handling class imbalance well for thin structures.  Focal Loss
    down-weights easy background pixels and focuses learning on hard examples
    such as road edges and shadow regions.

    Loss = alpha * DiceLoss + beta * FocalLoss

    Args:
        alpha (float): Weight applied to the Dice component.
        beta  (float): Weight applied to the Focal component.
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary")

    def forward(self, logits, targets):
        return self.alpha * self.dice(logits, targets) + self.beta * self.focal(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_metric(preds, labels, threshold=0.5):
    """
    Computes the Intersection over Union (IoU / Jaccard index) for a batch.

    Args:
        preds     (torch.Tensor): Raw logits from the model.
        labels    (torch.Tensor): Ground-truth binary mask tensor.
        threshold (float):        Probability threshold for binarisation.

    Returns:
        float: Scalar IoU value for the batch.
    """
    preds = torch.sigmoid(preds) > threshold
    labels = labels > 0.5
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    return ((intersection + 1e-6) / (union + 1e-6)).item()


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """
    Runs one full pass over the training dataloader with gradient accumulation.

    Args:
        loader    (DataLoader):       Training data loader.
        model     (nn.Module):        Model being trained.
        optimizer (torch.optim):      Optimiser instance.
        loss_fn   (nn.Module):        Loss function.
        device    (torch.device):     Compute device.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    accumulation_steps = CONFIG.get("GRAD_ACCUMULATION_STEPS", 1)

    for i, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)

        predictions = model(data)
        loss = loss_fn(predictions, targets) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        loop.set_postfix(loss=loss.item() * accumulation_steps)

    return running_loss / len(loader)


def evaluate(loader, model, loss_fn, device):
    """
    Evaluates the model on a validation dataloader.

    Args:
        loader   (DataLoader):   Validation data loader.
        model    (nn.Module):    Model in eval mode.
        loss_fn  (nn.Module):    Loss function.
        device   (torch.device): Compute device.

    Returns:
        tuple: (mean_val_loss, mean_val_iou)
    """
    model.eval()
    val_loss, val_iou = 0.0, 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation")
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            val_loss += loss_fn(predictions, targets).item()
            iou = iou_metric(predictions, targets)
            val_iou += iou
            loop.set_postfix(val_iou=iou)

    return val_loss / len(loader), val_iou / len(loader)


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def find_hard_samples(model, dataset, device, top_k_percent=0.2):
    """
    Identifies the hardest training samples by lowest per-image IoU.

    Iterates over the entire training dataset, computes IoU for each image,
    then returns the indices of the bottom top_k_percent fraction.

    Args:
        model         (nn.Module):    Trained model (set to eval mode internally).
        dataset       (Dataset):      Full training dataset.
        device        (torch.device): Compute device.
        top_k_percent (float):        Fraction of samples to select (0.0–1.0).

    Returns:
        np.ndarray: Array of dataset indices corresponding to hard samples.
    """
    print("Mining hard negatives (computing per-image IoU on training set) ...")
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["BATCH_SIZE"] * 2,
        shuffle=False,
        num_workers=4
    )

    ious = []
    indices = []

    with torch.no_grad():
        for i, (data, targets) in enumerate(tqdm(loader)):
            data, targets = data.to(device), targets.to(device)
            preds_prob = torch.sigmoid(model(data)) > 0.5
            start_idx = i * loader.batch_size

            for j in range(data.size(0)):
                p = preds_prob[j]
                t = targets[j]
                intersection = (p & t.byte()).float().sum()
                union = (p | t.byte()).float().sum()
                score = ((intersection + 1e-6) / (union + 1e-6)).item()
                ious.append(score)
                indices.append(start_idx + j)

    ious = np.array(ious)
    sorted_idx = np.argsort(ious)  # Ascending: lowest IoU first.
    n_hard = int(len(dataset) * top_k_percent)
    hard_indices = sorted_idx[:n_hard]

    print(f"Identified {len(hard_indices)} hard samples. Avg IoU: {np.mean(ious[hard_indices]):.4f}")
    return hard_indices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the V4 road segmentation model.")
    parser.add_argument("--resume",      action="store_true", help="Resume from checkpoint.")
    parser.add_argument("--hard-mining", action="store_true", help="Run hard negative mining after main training.")
    args = parser.parse_args()

    print("--- V4 Training Pipeline (EfficientNet-B4 + U-Net++) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(CONFIG["MODEL_SAVE_PATH"]), exist_ok=True)

    img_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], "images")
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Processed data not found at {img_dir}. Run preprocess.py first.")
        return

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    train_files, val_files = train_test_split(
        all_files,
        test_size=CONFIG["VALIDATION_SPLIT"],
        random_state=42
    )

    train_dataset = RoadSegmentationDatasetV4(img_dir, mask_dir, train_files, get_transforms(train=True))
    val_dataset   = RoadSegmentationDatasetV4(img_dir, mask_dir, val_files,   get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

    model     = create_model().to(device)
    loss_fn   = ComboLoss(alpha=0.5, beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["SCHEDULER_T_MAX"])

    start_epoch = 0
    best_iou = 0.0

    if args.resume and os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        ckpt = torch.load(CONFIG["CHECKPOINT_PATH"])
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_iou = ckpt["best_val_iou"]
        print(f"Resumed from epoch {start_epoch}. Best IoU so far: {best_iou:.4f}")

    log_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_iou", "lr"])

    # --- Phase 1: Main training ---
    print(f"Starting main training ({CONFIG['NUM_EPOCHS']} epochs) ...")
    for epoch in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['NUM_EPOCHS']}")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"Saved best model (IoU: {best_iou:.4f})")

        torch.save({
            "state_dict":    model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "epoch":         epoch + 1,
            "best_val_iou":  best_iou,
        }, CONFIG["CHECKPOINT_PATH"])

        new_row = pd.DataFrame([{
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_iou": val_iou, "lr": lr
        }])
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        log_df.to_csv(CONFIG["LOG_FILE"], index=False)

    # --- Phase 2: Hard negative mining (optional) ---
    if args.hard_mining:
        print("\n--- Hard Negative Mining Phase ---")
        hard_indices = find_hard_samples(model, train_dataset, device, top_k_percent=0.2)
        hard_dataset = Subset(train_dataset, hard_indices)
        hard_loader  = DataLoader(hard_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=8)

        ft_lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group["lr"] = ft_lr

        extra_epochs = 10
        print(f"Fine-tuning on {len(hard_dataset)} hard samples for {extra_epochs} epochs (LR: {ft_lr}) ...")

        for i in range(extra_epochs):
            epoch = CONFIG["NUM_EPOCHS"] + i
            print(f"\nHard Mining Epoch {i + 1}/{extra_epochs} (global epoch {epoch + 1})")
            train_loss = train_one_epoch(hard_loader, model, optimizer, loss_fn, device)
            val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
            print(f"HM Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
                print(f"Saved best model (IoU: {best_iou:.4f})")

    print("\nV4 training complete.")


if __name__ == "__main__":
    main()
