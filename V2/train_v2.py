
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Import V2 modules
from model_v2 import create_model
from dataset_v2 import RoadSegmentationDataset, get_transforms

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": 'data/processed/train',
    "MODEL_SAVE_PATH": 'weights/best_model_v2.pth',
    "LOG_FILE": 'training_log_v2.csv',
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "BATCH_SIZE": 16, # Adjust based on Colab GPU memory
    "NUM_EPOCHS": 75,
    "VALIDATION_SPLIT": 0.15,
    "ENCODER": "resnet34",
    "PRETRAINED": "imagenet",
    "SCHEDULER_T_MAX": 75, # Corresponds to the number of epochs
}

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

def main():
    """Main function to orchestrate the V2 training process."""
    print("Starting V2 training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    image_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    all_files = sorted(os.listdir(image_dir))
    train_files, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)

    train_dataset = RoadSegmentationDataset(image_dir, mask_dir, train_files, get_transforms(train=True))
    val_dataset = RoadSegmentationDataset(image_dir, mask_dir, val_files, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model, Loss, Optimizer, Scheduler ---
    model = create_model().to(device)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["SCHEDULER_T_MAX"])

    # --- Logging Setup ---
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'learning_rate'])

    # --- Training Loop ---
    best_val_iou = -1.0
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.6f}")

        # Log metrics
        new_log = pd.DataFrame([{'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_iou': val_iou, 'learning_rate': current_lr}])
        log_df = pd.concat([log_df, new_log], ignore_index=True)
        log_df.to_csv(CONFIG["LOG_FILE"], index=False)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"  -> New best model saved to {CONFIG['MODEL_SAVE_PATH']} (IoU: {best_val_iou:.4f})")

    print(f"\nTraining finished. Best Val IoU: {best_val_iou:.4f}")

if __name__ == '__main__':
    main()
