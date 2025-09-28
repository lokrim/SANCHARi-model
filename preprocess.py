
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

# Configuration
RAW_DATA_DIR = 'data/raw/train'
PROCESSED_DATA_DIR = 'data/processed/train'
IMG_SIZE = 1024
PATCH_SIZE = 256
NUM_PATCHES_PER_DIM = IMG_SIZE // PATCH_SIZE

def tile_image(img_path, mask_path, output_dir):
    """
    Tiles a single image and its corresponding mask into smaller patches.
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Read image and mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        return
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}. Skipping.")
        return

    # Get the base filename
    base_filename = os.path.splitext(os.path.basename(img_path))[0].replace('_sat', '')

    # Generate patches
    patch_count = 0
    for i in range(NUM_PATCHES_PER_DIM):
        for j in range(NUM_PATCHES_PER_DIM):
            # Define patch coordinates
            y_start, y_end = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x_start, x_end = j * PATCH_SIZE, (j + 1) * PATCH_SIZE

            # Extract patches
            img_patch = img[y_start:y_end, x_start:x_end]
            mask_patch = mask[y_start:y_end, x_start:x_end]

            # Save patches
            img_patch_path = os.path.join(output_dir, 'images', f'{base_filename}_patch_{patch_count}.jpg')
            mask_patch_path = os.path.join(output_dir, 'masks', f'{base_filename}_patch_{patch_count}.png')

            cv2.imwrite(img_patch_path, img_patch)
            cv2.imwrite(mask_patch_path, mask_patch)

            patch_count += 1

def main():
    """
    Main function to preprocess the entire dataset.
    """
    print("Starting preprocessing...")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")

    # Get all satellite image paths
    sat_image_paths = glob.glob(os.path.join(RAW_DATA_DIR, '*_sat.jpg'))
    
    if not sat_image_paths:
        print(f"Error: No satellite images found in {RAW_DATA_DIR}. Make sure the dataset is downloaded and extracted correctly.")
        return

    print(f"Found {len(sat_image_paths)} satellite images to process.")

    # Process each image
    for img_path in tqdm(sat_image_paths, desc="Tiling images and masks"):
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for image {img_path}. Skipping.")
            continue
        tile_image(img_path, mask_path, PROCESSED_DATA_DIR)

    print("Preprocessing complete.")
    print(f"Tiled images and masks are saved in {PROCESSED_DATA_DIR}")

if __name__ == '__main__':
    main()
