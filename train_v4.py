
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import numpy as np

# Import V4 modules
from model_v4 import create_model_v4
from dataset_v4 import RoadSegmentationDatasetV4, get_transforms_v4

# --- Configuration ---
CONFIG = {
    # Paths (V4 uses processed_v4 folder)
    "PROCESSED_DATA_DIR": 'data/processed_v4/train', 
    "MODEL_SAVE_PATH": 'weights/best_model_v4.pth',     
    "CHECKPOINT_PATH": 'checkpoint_v4.pth',
    "LOG_FILE": 'training_log_v4.csv',
    
    # Hyperparameters for V4 (Adjusted for 4090)
    "LEARNING_RATE": 5e-4,         
    "WEIGHT_DECAY": 1e-4,          
    "BATCH_SIZE": 8,               # Reduced from 16 to 8 to fix OOM on RTX 4090
    "GRAD_ACCUMULATION_STEPS": 2,  # maintain effective batch size of 16
    "NUM_EPOCHS": 50,
    "SCHEDULER_T_MAX": 50,

    "VALIDATION_SPLIT": 0.15,
}

# --- Combo Loss (Dice + Focal) ---
class ComboLoss(nn.Module):
    """
    Weighted combination of Dice Loss and Focal Loss.
    Loss = alpha * Dice + beta * Focal
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary') # SMP Focal expects logits by default

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.alpha * dice_loss + self.beta * focal_loss

def iou_metric(preds, labels, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold)
    labels = labels > 0.5 # Ensure boolean
    
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler=None):
    model.train()
    loop = tqdm(loader, desc='Training')
    running_loss = 0.0
    
    accumulation_steps = CONFIG.get("GRAD_ACCUMULATION_STEPS", 1)
    
    for i, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        
        # Mixed Precision (Optional, but good for VRAM)
        # For now, standard FP32 as user didn't request AMP, but straightforward to add.
        # We stick to FP32 for stability unless requested.
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps # Scale back for logging
        loop.set_postfix(loss=loss.item() * accumulation_steps)
        
    return running_loss / len(loader)

def evaluate(loader, model, loss_fn, device):
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

def find_hard_samples(model, dataset, device, top_k_percent=0.2):
    """
    Identifies the 'hardest' samples (lowest IoU) in the training set.
    """
    print("Mininig Hard Negatives (Calculating IoU on Train Set)...")
    model.eval()
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=4) # Faster inference
    
    ious = []
    indices = []
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(tqdm(loader)):
            data, targets = data.to(device), targets.to(device)
            preds = model(data)
            
            # Calculate IoU per image in batch
            preds_prob = torch.sigmoid(preds) > 0.5
            start_idx = i * loader.batch_size
            
            for j in range(data.size(0)):
                # Scalar IoU for single image
                p = preds_prob[j]
                t = targets[j]
                intersection = (p & t.byte()).float().sum()
                union = (p | t.byte()).float().sum()
                score = (intersection + 1e-6) / (union + 1e-6)
                ious.append(score.item())
                indices.append(start_idx + j)
                
    # Select indices with lowest IoU
    ious = np.array(ious)
    sorted_idx = np.argsort(ious) # Ascending order (lowest first)
    n_hard = int(len(dataset) * top_k_percent)
    hard_indices = sorted_idx[:n_hard]
    
    avg_tough_iou = np.mean(ious[hard_indices])
    print(f"Identified {len(hard_indices)} hard samples. Avg IoU: {avg_tough_iou:.4f}")
    
    return hard_indices

def main():
    parser = argparse.ArgumentParser(description="Train V4 Model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--hard-mining", action="store_true", help="Enable Hard Negative Mining phase after main training")
    args = parser.parse_args()

    print("--- V4 Training Pipeline (EfficientNet-B4 + U-Net++) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure weights directory exists
    model_save_dir = os.path.dirname(CONFIG["MODEL_SAVE_PATH"])
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Created directory: {model_save_dir}")

    # Directories
    img_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    
    if not os.path.exists(img_dir):
        print(f"Error: Processed data not found at {img_dir}. Run preprocess_v4.py first.")
        return

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    train_files, val_files = train_test_split(all_files, test_size=CONFIG["VALIDATION_SPLIT"], random_state=42)

    # Datasets
    train_dataset = RoadSegmentationDatasetV4(img_dir, mask_dir, train_files, get_transforms_v4(train=True))
    val_dataset = RoadSegmentationDatasetV4(img_dir, mask_dir, val_files, get_transforms_v4(train=False))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

    # Model Setup
    model = create_model_v4().to(device)
    loss_fn = ComboLoss(alpha=0.5, beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["SCHEDULER_T_MAX"])

    # Resume
    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        ckpt = torch.load(CONFIG["CHECKPOINT_PATH"])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_iou = ckpt['best_val_iou']
        print(f"Resumed from Epoch {start_epoch}, Best IoU: {best_iou:.4f}")

    # Logging
    log_dir = os.path.dirname(CONFIG["LOG_FILE"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'lr'])
    
    # ---------------- MAIN TRAINING PHASE ----------------
    print(f"Starting Main Training ({CONFIG['NUM_EPOCHS']} Epochs)...")
    for epoch in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save Best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"Saved Best Model (IoU: {best_iou:.4f})")
            
        # Checkpoint
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_iou': best_iou
        }, CONFIG["CHECKPOINT_PATH"])

    # ---------------- HARD NEGATIVE MINING PHASE ----------------
    if args.hard_mining:
        print("\n--- Starting Hard Negative Mining Phase ---")
        # 1. Identify Hard Samples
        hard_indices = find_hard_samples(model, train_dataset, device, top_k_percent=0.2)
        
        # 2. Create Subset
        hard_dataset = Subset(train_dataset, hard_indices)
        hard_loader = DataLoader(hard_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=8)
        
        # 3. Fine-tune for extra epochs
        # Lower LR for fine-tuning
        ft_lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = ft_lr
            
        extra_epochs = 10
        print(f"Fine-tuning on {len(hard_dataset)} hard samples for {extra_epochs} epochs (LR: {ft_lr})...")
        
        for i in range(extra_epochs):
            epoch = CONFIG["NUM_EPOCHS"] + i
            print(f"\nHard Mining Epoch {i+1}/{extra_epochs} (Global {epoch+1})")
            train_loss = train_one_epoch(hard_loader, model, optimizer, loss_fn, device)
            val_loss, val_iou = evaluate(val_loader, model, loss_fn, device) # Validate on full val set
            print(f"HM Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
                print(f"Saved Best Model (IoU: {best_iou:.4f})")

    print("\nV4 Training Complete.")

if __name__ == "__main__":
    main()
