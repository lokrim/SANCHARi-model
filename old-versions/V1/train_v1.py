
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm

from model import UNet
from dataset import RoadSegmentationDataset, get_transforms

# --- Configuration ---
PROCESSED_DATA_DIR = '../../data/processed/train'
MODEL_SAVE_PATH = '../../weights/best_model_v1.pth'

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 # Adjust based on your Colab GPU memory
NUM_EPOCHS = 25 # Start with a reasonable number, can be increased
VALIDATION_SPLIT = 0.15

def iou_metric(preds, labels, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) metric for a batch.
    Args:
        preds (torch.Tensor): Predicted logits from the model (B, 1, H, W).
        labels (torch.Tensor): Ground truth masks (B, 1, H, W).
        threshold (float): Threshold to binarize the predictions.
    Returns:
        float: The average IoU score for the batch.
    """
    # Apply sigmoid and threshold to get binary predictions
    preds = torch.sigmoid(preds) > threshold
    preds = preds.byte()
    labels = labels.byte()

    # Flatten
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Calculate intersection and union
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()

    # Calculate IoU, add a small epsilon to avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    loop = tqdm(loader, desc='Training')
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def evaluate(loader, model, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

            # Calculate IoU
            iou = iou_metric(predictions, targets)
            val_iou += iou
            loop.set_postfix(val_iou=iou)

    return val_loss / len(loader), val_iou / len(loader)

def main():
    """
    Main function to orchestrate the training process.
    """
    print("Starting training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    image_dir = os.path.join(PROCESSED_DATA_DIR, 'images')
    mask_dir = os.path.join(PROCESSED_DATA_DIR, 'masks')

    # Get list of image files and create a validation split
    image_files = sorted(os.listdir(image_dir))
    train_files, val_files = train_test_split(image_files, test_size=VALIDATION_SPLIT, random_state=42)

    # Create datasets
    train_dataset = RoadSegmentationDataset(
        image_dir=image_dir, mask_dir=mask_dir, transform=get_transforms(train=True)
    )
    # Manually set the images for the train_dataset
    train_dataset.images = train_files

    val_dataset = RoadSegmentationDataset(
        image_dir=image_dir, mask_dir=mask_dir, transform=get_transforms(train=False)
    )
    # Manually set the images for the val_dataset
    val_dataset.images = val_files

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=3, n_classes=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss() # Well-suited for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    best_val_iou = -1.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val IoU:    {val_iou:.4f}")

        # Save the model if it has the best validation IoU so far
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (IoU: {best_val_iou:.4f})")

    print("\nTraining finished.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
