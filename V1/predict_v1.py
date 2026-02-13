
import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

from model import UNet
from dataset import get_transforms

# --- Configuration ---
MODEL_PATH = 'weights/best_model_v1.pth'
PATCH_SIZE = 256
IMG_SIZE = 1024
NUM_PATCHES_PER_DIM = IMG_SIZE // PATCH_SIZE

def predict_single_image(model, image_path, device):
    """
    Runs prediction on a single 1024x1024 image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a blank canvas for the final mask
    full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Get the validation/test transforms
    transform = get_transforms(train=False)

    # Process the image in patches
    for i in range(NUM_PATCHES_PER_DIM):
        for j in range(NUM_PATCHES_PER_DIM):
            # Define patch coordinates
            y_start, y_end = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x_start, x_end = j * PATCH_SIZE, (j + 1) * PATCH_SIZE

            # Extract and transform the patch
            patch = img[y_start:y_end, x_start:x_end]
            transformed_patch = transform(image=patch)['image'].unsqueeze(0).to(device)

            # Run prediction
            with torch.no_grad():
                pred_logit = model(transformed_patch)
                pred_prob = torch.sigmoid(pred_logit)
                pred_mask = (pred_prob > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            # Place the predicted patch on the canvas
            full_mask[y_start:y_end, x_start:x_end] = pred_mask * 255

    return full_mask

def main():
    """
    Main function to run inference on a folder of images.
    """
    parser = argparse.ArgumentParser(description="Road Segmentation Inference")
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder containing input images.')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the folder to save predicted masks.')
    args = parser.parse_args()

    print("Starting inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return

    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    # --- Create Output Directory ---
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Process Images ---
    image_files = [f for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print(f"No images found in {args.input_folder}")
        return

    for img_name in tqdm(image_files, desc="Predicting masks"):
        img_path = os.path.join(args.input_folder, img_name)
        
        # Perform prediction
        predicted_mask = predict_single_image(model, img_path, device)

        if predicted_mask is not None:
            # Save the final mask
            output_path = os.path.join(args.output_folder, f"{os.path.splitext(img_name)[0]}_pred_mask.png")
            cv2.imwrite(output_path, predicted_mask)

    print(f"\nInference complete. Masks saved to {args.output_folder}")

if __name__ == '__main__':
    main()
