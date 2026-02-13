
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse

# Import V3 modules
from model_v3 import create_model_v3
from dataset_v3 import RoadSegmentationDatasetV3, get_transforms_v3

# --- Configuration ---
CONFIG = {
    # Paths
    "PROCESSED_DATA_DIR": 'data/processed/train', # Directory containing tiled images/masks
    "MODEL_SAVE_PATH": 'weights/best_model_v3.pth',     # Path to save the best model weights
    "CHECKPOINT_PATH": 'checkpoint_v3.pth', # Path for resuming
    "LOG_FILE": 'training_log_v3.csv',            # Path to save training metrics
    
    # Hyperparameters
    "LEARNING_RATE": 1e-4,     # Initial learning rate
    "WEIGHT_DECAY": 1e-5,      # Regularization to prevent overfitting
    "BATCH_SIZE": 16,          # Adjust based on GPU VRAM (16 is good for 16GB VRAM)
    "NUM_EPOCHS": 50,          # Total training epochs
    "SCHEDULER_T_MAX": 75, # For CosineAnnealingLR

    # Dataset config
    "VALIDATION_SPLIT": 0.15, # Fraction of data to use for validation (renamed from VAL_SPLIT to match original)
    "INPUT_SHAPE": (256, 256), # Patch size (H, W) matches PATCH_SIZE in preprocess
}

# --- Custom Dice Loss ---
class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Optimizes the Overlap (Intersection over Union related metric).
    Range: 0 (perfect overlap) to 1 (no overlap).
    """
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

def iou_metric(preds, labels, threshold=0.5):
    """Calculates Intersection over Union (IoU) for a batch."""
    preds = torch.sigmoid(preds) > threshold
    preds, labels = preds.byte(), labels.byte()
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """Trains the model for one epoch."""
    model.train()
    loop = tqdm(loader, desc='Training')
    running_loss = 0.0
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)

def evaluate(loader, model, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
            iou = iou_metric(predictions, targets)
            val_iou += iou
            loop.set_postfix(val_iou=iou)
    return val_loss / len(loader), val_iou / len(loader)

def save_checkpoint(state, filename):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_iou', -1.0)

def main():
    parser = argparse.ArgumentParser(description="Train V3 Model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    print("Starting V3 training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    image_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    train_files, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)

    train_dataset = RoadSegmentationDatasetV3(image_dir, mask_dir, train_files, get_transforms_v3(train=True))
    val_dataset = RoadSegmentationDatasetV3(image_dir, mask_dir, val_files, get_transforms_v3(train=False))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Model, Loss, Optimizer, Scheduler ---
    model = create_model_v3().to(device)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["SCHEDULER_T_MAX"])

    start_epoch = 0
    best_val_iou = -1.0
    
    # --- Resume Logic ---
    if args.resume and os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=device)
        start_epoch, best_val_iou = load_checkpoint(checkpoint, model, optimizer)
        # Adjust scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch} with Best IoU: {best_val_iou:.4f}")
    
    # --- Logging Setup ---
    if args.resume and os.path.exists(CONFIG["LOG_FILE"]):
        log_df = pd.read_csv(CONFIG["LOG_FILE"])
    else:
        log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'learning_rate'])

    # --- Training Loop ---
    for epoch in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.6f}")

        # Log metrics
        new_log = pd.DataFrame([{'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_iou': val_iou, 'learning_rate': current_lr}])
        log_df = pd.concat([log_df, new_log], ignore_index=True)
        log_df.to_csv(CONFIG["LOG_FILE"], index=False)

        # Save Checkpoint (Every Epoch for safety)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_iou': best_val_iou, # Keep the global best
        }
        save_checkpoint(checkpoint, CONFIG["CHECKPOINT_PATH"])

        # Save Best Model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            # Update best IoU in checkpoint meta-data if needed, but 'best_val_iou' variable tracks it
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"  -> New best model saved to {CONFIG['MODEL_SAVE_PATH']} (IoU: {best_val_iou:.4f})")

    print(f"\nTraining finished. Best Val IoU: {best_val_iou:.4f}")

if __name__ == '__main__':
    main()
