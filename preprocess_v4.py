
import os
import glob
import zipfile
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import subprocess
import argparse
import shutil

# --- Configuration ---
DATASET_NAME = "balraj98/deepglobe-road-extraction-dataset"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed_v4/train" # V4 specific directory
TILE_SIZE = 512  # V4: Larger Context (512x512)
STRIDE = 256     # V4: 50% Overlap (Stride = Size // 2)

def download_dataset():
    """
    Downloads and unzips the DeepGlobe Road Extraction dataset from Kaggle.
    Skips if data is already present.
    """
    print(f"Downloading {DATASET_NAME}...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Check if data already exists to avoid re-downloading
    if os.path.exists(os.path.join(RAW_DATA_DIR, "train")):
        print("Dataset seems to be already present in data/raw. Skipping download.")
        return

    try:
        # Use Kaggle CLI (must be installed and configured with kaggle.json)
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", RAW_DATA_DIR], check=True)
        
        # Unzip
        zip_path = os.path.join(RAW_DATA_DIR, "deepglobe-road-extraction-dataset.zip")
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        
        # Cleanup zip
        os.remove(zip_path)
        print("Download and extraction complete.")
        
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install it with 'pip install kaggle' and set up your ~/.kaggle/kaggle.json.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def tile_image_and_mask(image_path, mask_path, dest_dir, tile_size=512, stride=256):
    """
    Tiles a large satellite image and its corresponding mask into smaller patches
    using a sliding window approach with overlap.
    
    Args:
        image_path (str): Path to source image.
        mask_path (str): Path to source binary mask.
        dest_dir (str): Directory to save processed tiles.
        tile_size (int): Size of the square tile (e.g., 512).
        stride (int): Step size for sliding window (e.g., 256 for 50% overlap).
    """
    base_name = os.path.basename(image_path).split('.')[0]
    
    # Read Image using Rasterio (handles GeoTIFFs robustly)
    with rasterio.open(image_path) as src:
        image = src.read()
        # Rasterio reads (Bands, H, W) -> Convert to (H, W, Bands) for OpenCV/Saving
        image = np.moveaxis(image, 0, -1)
        # Ensure RGB (Keep first 3 bands if multispectral)
        image = image[:, :, :3]
    
    # Read Mask using OpenCV (Grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found for {image_path}")
        return

    h, w, _ = image.shape
    
    # Padding to ensure we can cover the edges with the window
    # Unlike V3 (exact multiples), here we want to ensure the last window fits.
    # Simple strategy: Pad to multiple of stride, then ensure size fits?
    # Better: Pad so that (W - Size) % Stride == 0? 
    # Use copyMakeBorder with REFLECT to handle boundaries safely.
    
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Actually, for sliding window with overlap, we might need more padding if the last step
    # doesn't align. But simplest is to just pad to allow full tile extraction.
    # Let's pad enough to cover.
    
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    h_padded, w_padded, _ = image.shape
    
    idx = 0
    # Sliding Window Loop
    for y in range(0, h_padded - tile_size + 1, stride):
        for x in range(0, w_padded - tile_size + 1, stride):
            # Extract tile
            img_tile = image[y:y+tile_size, x:x+tile_size]
            mask_tile = mask[y:y+tile_size, x:x+tile_size]
            
            # Save
            # Naming convention includes x, y to potentially reconstruct if needed, 
            # but simpler just index for unique filenames.
            out_name = f"{base_name}_{idx}"
            
            # Create subdirs "images" and "masks" inside dest_dir? 
            # V3 put them all in dest_dir (flat). 
            # Let's follow V3 convention but maybe organize better?
            # V3 dataset_v3 expected 'images' and 'masks' subdirs?
            # Checking V3 code: 
            # image_dir = os.path.join(CONFIG["PROCESSED_DATA_DIR"], 'images')
            # So V3 preprocess MUST HAVE created subdirs.
            # Let's check preprocess_v3 code again...
            # Ah, V3 preprocess_v3 code: 
            # cv2.imwrite(os.path.join(dest_dir, f"{out_name}.jpg"), ...)
            # Wait, did V3 preprocess create subdirs?
            # "os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)"
            # It saved to PROCESSED_DATA_DIR directly.
            # But train_v3.py looks for "PROCESSED_DATA_DIR/images".
            # This implies V3 IS BROKEN or I misread.
            # Let's fix this in V4. We will create 'images' and 'masks' subfolders.
            
            img_out_dir = os.path.join(dest_dir, "images")
            mask_out_dir = os.path.join(dest_dir, "masks")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(img_out_dir, f"{out_name}.jpg"), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_out_dir, f"{out_name}.png"), mask_tile) # Save mask as png
            idx += 1

def main():
    parser = argparse.ArgumentParser(description="V4 Preprocessing: 512x512 with Overlap")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    args = parser.parse_args()

    print("--- V4 Preprocessing Pipeline ---")
    print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Stride: {STRIDE} (50% Overlap)")
    
    # 1. Download Data
    if args.download:
        download_dataset()
    else:
        print("Skipping download (use --download to force). Checking local data...")
    
    # 2. Setup Directories
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Cleaning existing V4 processed data at {PROCESSED_DATA_DIR}...")
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 3. Locate Images
    search_path = os.path.join(RAW_DATA_DIR, "**", "*_sat.jpg")
    sat_files = glob.glob(search_path, recursive=True)
    
    if not sat_files:
        print("No satellite images found. Please check data/raw structure or run with --download.")
        return
        
    print(f"Found {len(sat_files)} source images. Starting tiling...")
    
    # 4. Process Each Image
    for img_path in tqdm(sat_files):
        # Infer mask path: _sat.jpg -> _mask.png
        # Try multiple common patterns just in case
        potential_masks = [
            img_path.replace('_sat.jpg', '_mask.png'),
            img_path.replace('sat.jpg', 'mask.png'),
            img_path.replace('_sat.jpg', '_mask.jpg') # Some might be jpg
        ]
        
        mask_path = None
        for p in potential_masks:
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            tile_image_and_mask(img_path, mask_path, PROCESSED_DATA_DIR, TILE_SIZE, STRIDE)
        else:
            # Fallback for some datasets where mask might be missing
            pass
            
    print(f"Preprocessing complete. V4 Tiles saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
