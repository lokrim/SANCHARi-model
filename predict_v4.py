
import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
from skimage.morphology import skeletonize, remove_small_objects, closing, disk
import rasterio

# Import V4 model
from model_v4 import create_model_v4
from postprocess_v4 import apply_advanced_postprocessing

# --- Configuration ---
TEST_IMAGES_DIR = "test-images"       # Directory containing input images for batch inference
OUTPUT_DIR = "predicted/predictedv4"            # Directory to save output predictions (V4)
MODEL_PATH = "weights/best_model_v4.pth" # Path to trained model weights
PATCH_SIZE = 512                      # V4: Larger Patch Size (512x512)
STRIDE = 256                          # V4: 50% Overlap (Stride 256)

def predict_sliding_window(large_image, model, device):
    """
    Performs sliding window inference on a large image.
    
    Args:
        large_image (np.array): Input image (H, W, 3) in RGB.
        model (torch.nn.Module): Trained model.
        device (torch.device): CUDA or CPU.
        
    Returns:
        np.array: Probability map (H, W) with values 0.0-1.0.
    """
    h, w, _ = large_image.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Padding to fit patches exactly
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    
    # Reflect padding to minimize edge artifacts
    padded_image = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    h_padded, w_padded, _ = padded_image.shape
    
    for y in range(0, h_padded - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_padded - PATCH_SIZE + 1, STRIDE):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Preprocess patch: (H,W,C) -> (C,H,W), Normalize to 0-1, Add Batch Dim
            img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            
            # Normalize (match training) - V4 uses ImageNet normalization (EfficientNet requirement)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            # --- 4-Way Test Time Augmentation (TTA) ---
            # 1. Original
            with torch.no_grad():
                probs = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
            
            # 2. Horizontal Flip
            with torch.no_grad():
                probs_h = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3]).squeeze().cpu().numpy()
            
            # 3. Vertical Flip
            with torch.no_grad():
                probs_v = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2]).squeeze().cpu().numpy()
            
            # 4. Rotate 90
            with torch.no_grad():
                probs_rot = torch.rot90(torch.sigmoid(model(torch.rot90(img_tensor, 1, [2, 3]))), -1, [2, 3]).squeeze().cpu().numpy()
            
            # Average predictions
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            
            # Accumulate
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_avg
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Normalize by overlap count
    count_map[count_map == 0] = 1
    prob_map /= count_map
    
    # Crop padding
    prob_map = prob_map[:h, :w]
    
    return prob_map

def main():
    parser = argparse.ArgumentParser(description="Predict on test images using V4 model")
    parser.add_argument("--input", default=TEST_IMAGES_DIR, help="Input directory of images")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model (V4 Architecture)
    model = create_model_v4().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"V4 Model weights loaded from {MODEL_PATH}.")
    else:
        print(f"Error: Weights not found at {MODEL_PATH}")
        return
    model.eval()
    
    # Setup Output Directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get Test Images
    test_images = glob.glob(os.path.join(args.input, "*.jpg"))
    if not test_images:
        print(f"No images found in {args.input}")
        return
        
    print(f"Found {len(test_images)} test images. Starting V4 Inference...")
    
    for img_path in tqdm(test_images):
        base_name = os.path.basename(img_path).split('.')[0]
        
        # Read Image (OpenCV reads BGR)
        image = cv2.imread(img_path)
        # Convert to RGB (Model expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        # Inference
        prob_map = predict_sliding_window(image_rgb, model, device)


# ...

        # --- Post-Processing (Advanced V4) ---
        binary_mask, skeleton = apply_advanced_postprocessing(prob_map, threshold=0.45)
        
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        prob_uint8 = (prob_map * 255).astype(np.uint8)

        # Save Inputs & Outputs (Debug Style)
        # 1. Original Input
        cv2.imwrite(os.path.join(args.output, f"{base_name}_input.jpg"), image) # Save BGR original
        
        # 2. Probability Map
        cv2.imwrite(os.path.join(args.output, f"{base_name}_prob.png"), prob_uint8)
        
        # 3. Binary Mask
        cv2.imwrite(os.path.join(args.output, f"{base_name}_mask.png"), mask_uint8)
        
        # 4. Skeleton
        cv2.imwrite(os.path.join(args.output, f"{base_name}_skeleton.png"), skeleton_uint8)
        
        # 5. Overlay
        overlay = image.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255] # Red where mask is
        combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_overlay.jpg"), combined)

    print(f"V4 Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
