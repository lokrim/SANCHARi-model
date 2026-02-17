
import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

# Import V2 modules
from model_v2 import create_model
from dataset_v2 import get_transforms

# --- Configuration ---
CONFIG = {
    "MODEL_PATH": 'weights/best_model_v2.pth',
    "OPTIMAL_THRESHOLD": 0.4,
    "PATCH_SIZE": 256,
    "IMG_SIZE": 1024,
}

NUM_PATCHES_PER_DIM = CONFIG["IMG_SIZE"] // CONFIG["PATCH_SIZE"]

def predict_single_image(model, image_path, device, transform):
    """
    Runs prediction on a single 1024x1024 image using a tiling strategy.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    full_mask = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), dtype=np.uint8)

    for i in range(NUM_PATCHES_PER_DIM):
        for j in range(NUM_PATCHES_PER_DIM):
            y_start, y_end = i * CONFIG["PATCH_SIZE"], (i + 1) * CONFIG["PATCH_SIZE"]
            x_start, x_end = j * CONFIG["PATCH_SIZE"], (j + 1) * CONFIG["PATCH_SIZE"]

            patch = img[y_start:y_end, x_start:x_end]
            transformed_patch = transform(image=patch)['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                pred_logit = model(transformed_patch)
                pred_prob = torch.sigmoid(pred_logit)
                pred_mask = (pred_prob > CONFIG["OPTIMAL_THRESHOLD"]).squeeze().cpu().numpy().astype(np.uint8)

            full_mask[y_start:y_end, x_start:x_end] = pred_mask * 255

    return full_mask

def main():
    """
    Main function to run batch inference on a folder of images.
    """
    parser = argparse.ArgumentParser(description="V2 Road Segmentation Inference")
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder of input images.')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to save predicted masks.')
    args = parser.parse_args()

    print("Starting V2 inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"Error: Model file not found at {CONFIG['MODEL_PATH']}. Please run train_v2.py first.")
        return

    model = create_model().to(device)
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
    model.eval()
    print(f"Model loaded from {CONFIG['MODEL_PATH']}")

    # --- Get Transforms and Create Output Directory ---
    inference_transform = get_transforms(train=False)
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Process Images ---
    image_files = [f for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print(f"No images found in {args.input_folder}")
        return

    for img_name in tqdm(image_files, desc="Predicting masks"):
        img_path = os.path.join(args.input_folder, img_name)
        predicted_mask = predict_single_image(model, img_path, device, inference_transform)

        if predicted_mask is not None:
            output_path = os.path.join(args.output_folder, f"{os.path.splitext(img_name)[0]}_pred_mask_v2.png")
            cv2.imwrite(output_path, predicted_mask)

    print(f"\nInference complete. Masks saved to {args.output_folder}")

if __name__ == '__main__':
    main()
