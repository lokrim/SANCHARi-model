import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
from skimage.morphology import skeletonize, remove_small_objects, closing, disk

# Import V3 model
from model_v3 import create_model_v3

# --- Configuration ---
TEST_IMAGES_DIR = "test-images"       # Directory containing input images for batch inference
OUTPUT_DIR = "predictedv3"            # Directory to save output predictions
MODEL_PATH = "weights/best_model_v3.pth" # Path to trained model weights
PATCH_SIZE = 256                      # Patch size used during training
STRIDE = 128                          # Stride for sliding window (overlap helps continuity)

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
    # If pad_h or pad_w is PATCH_SIZE, it means h % PATCH_SIZE is 0, so no padding needed actually.
    # The modulo trick handles this correctly: (256 - 0) % 256 = 0.
    
    padded_image = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    h_padded, w_padded, _ = padded_image.shape
    
    for y in range(0, h_padded - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_padded - PATCH_SIZE + 1, STRIDE):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Preprocess patch: (H,W,C) -> (C,H,W), Normalize to 0-1, Add Batch Dim
            img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            
            # Normalize (match training) - V3 training uses ImageNet normalization
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
    parser = argparse.ArgumentParser(description="Predict on test images using V3 model")
    parser.add_argument("--input", default=TEST_IMAGES_DIR, help="Input directory of images")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = create_model_v3().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded.")
    else:
        print(f"Error: Weights not found at {MODEL_PATH}")
        return
    model.eval()
    
    # Setup Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get Test Images
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    if not test_images:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return
        
    print(f"Found {len(test_images)} test images.")
    
    for img_path in tqdm(test_images):
        base_name = os.path.basename(img_path).split('.')[0]
        
        # Read Image (OpenCV reads BGR)
        image = cv2.imread(img_path)
        # Convert to RGB (Model expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference
        prob_map = predict_sliding_window(image_rgb, model, device)
        
        # --- Post-Processing ---
        # 1. Thresholding
        binary_mask = prob_map > 0.45 
        
        # 2. Morphology: Remove Small Objects
        try:
            binary_mask = remove_small_objects(binary_mask, max_size=100)
        except TypeError:
             # Fallback for older skimage versions
            binary_mask = remove_small_objects(binary_mask, min_size=100)
            
        # 3. Morphology: Closing (Fill gaps)
        binary_mask = closing(binary_mask, footprint=disk(3))
        skeleton = skeletonize(binary_mask)
        skeleton_uint8 = skeleton.astype(np.uint8) * 255
        mask_uint8 = binary_mask.astype(np.uint8) * 255
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
        
        # 5. Overlay (Optional but helpful)
        # Create a red overlay for the road network on the original image
        overlay = image.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255] # Red where mask is
        # Blend
        combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_overlay.jpg"), combined)

    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
