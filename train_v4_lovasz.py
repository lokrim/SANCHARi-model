
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
import numpy as np

# Import V4 modules
from model_v4 import create_model_v4
from dataset_v4 import RoadSegmentationDatasetV4, get_transforms_v4

# --- Configuration ---
CONFIG = {
    # Paths
    "PROCESSED_DATA_DIR": 'data/processed_v4/train', 
    "PRETRAINED_MODEL_PATH": 'weights/best_model_v4.pth', # Start from best V4 model
    "MODEL_SAVE_PATH": 'weights/best_model_v4_lovasz.pth',
    "LOG_FILE": 'training_log_v4_lovasz.csv',
    
    # Fine-tuning Hyperparameters
    "LEARNING_RATE": 1e-5,          # Very low LR for fine-tuning
    "WEIGHT_DECAY": 1e-4,          
    "BATCH_SIZE": 8,               
    "GRAD_ACCUMULATION_STEPS": 2, 
    "NUM_EPOCHS": 20,              # Short fine-tuning phase
    "SCHEDULER_T_MAX": 20,

    "VALIDATION_SPLIT": 0.15,
}

def iou_metric(preds, labels, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold)
    labels = labels > 0.5
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc='Fine-tuning (Lovász)')
    running_loss = 0.0
    
    accumulation_steps = CONFIG.get("GRAD_ACCUMULATION_STEPS", 1)
    
    for i, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
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

def main():
    print("--- V4 Lovász Fine-tuning Pipeline ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Directories
    img_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
    mask_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'masks')
    
    if not os.path.exists(img_dir):
        print(f"Error: Data not found at {img_dir}")
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
    
    # Load Pretrained Weights
    if os.path.exists(CONFIG["PRETRAINED_MODEL_PATH"]):
        print(f"Loading weights from {CONFIG['PRETRAINED_MODEL_PATH']}...")
        model.load_state_dict(torch.load(CONFIG["PRETRAINED_MODEL_PATH"], map_location=device))
    else:
        print(f"Error: Pretrained weights not found at {CONFIG['PRETRAINED_MODEL_PATH']}")
        # We allow continuing (maybe training from scratch with Lovasz? No, unstable.)
        return

    # Loss Function: Lovász-Hinge (Binary)
    # mode='binary', per_image=True is often better for batch processing
    loss_fn = smp.losses.LovaszLoss(mode='binary', per_image=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["SCHEDULER_T_MAX"])

    # Logging
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'lr'])
    
    best_iou = 0.0
    
    # Evaluate Initial Performance
    print("Evaluating initial performance...")
    _, initial_iou = evaluate(val_loader, model, loss_fn, device)
    print(f"Initial Val IoU: {initial_iou:.4f}")
    best_iou = initial_iou

    print(f"Starting Fine-tuning ({CONFIG['NUM_EPOCHS']} Epochs)...")
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_iou = evaluate(val_loader, model, loss_fn, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss (Lovasz): {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save Best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"Saved Best Lovász Model (IoU: {best_iou:.4f})")
            
        # Log
        new_row = pd.DataFrame([{'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_iou': val_iou, 'lr': lr}])
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        log_df.to_csv(CONFIG["LOG_FILE"], index=False)

    print("\nLovász Fine-tuning Complete.")

if __name__ == "__main__":
    main()
