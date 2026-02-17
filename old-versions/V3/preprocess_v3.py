import os
import glob
import zipfile
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import subprocess

# --- Configuration ---
DATASET_NAME = "balraj98/deepglobe-road-extraction-dataset"
RAW_DATA_DIR = "../../data/raw"
PROCESSED_DATA_DIR = "../../data/processed/train"
TILE_SIZE = 256  # Size of identifying tiles (256x256)
OVERLAP = 0      # Overlap between tiles (0 for training data to avoid duplicates/leakage)

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

def tile_image_and_mask(image_path, mask_path, dest_dir, tile_size=256):
    """
    Tiles a large satellite image and its corresponding mask into smaller patches.
    
    Args:
        image_path (str): Path to source image.
        mask_path (str): Path to source binary mask.
        dest_dir (str): Directory to save processed tiles.
        tile_size (int): Size of the square tile.
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
    
    # Padding to ensure exact multiples of tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Reflect padding helps avoid border artifacts
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    h_padded, w_padded, _ = image.shape
    
    idx = 0
    for y in range(0, h_padded, tile_size):
        for x in range(0, w_padded, tile_size):
            # Extract tile
            img_tile = image[y:y+tile_size, x:x+tile_size]
            mask_tile = mask[y:y+tile_size, x:x+tile_size]
            
            # Save only if the tile has valid data (optional check, but good for cleaning)
            # For road segmentation, we might want to keep empty tiles to learn "background"
            # But we can skip tiles that are purely black padding if we changed logic.
            # Here we save everything.
            
            out_name = f"{base_name}_{idx}"
            cv2.imwrite(os.path.join(dest_dir, f"{out_name}.jpg"), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dest_dir, f"{out_name}_mask.png"), mask_tile)
            idx += 1

def main():
    print("--- V3 Preprocessing Pipeline ---")
    
    # 1. Download Data
    download_dataset()
    
    # 2. Setup Directories
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 3. Locate Images
    # Search for files ending in _sat.jpg (DeepGlobe convention)
    search_path = os.path.join(RAW_DATA_DIR, "**", "*_sat.jpg")
    sat_files = glob.glob(search_path, recursive=True)
    
    if not sat_files:
        print("No satellite images found. Please check data/raw structure.")
        return
        
    print(f"Found {len(sat_files)} source images. Starting tiling...")
    
    # 4. Process Each Image
    for img_path in tqdm(sat_files):
        # Infer mask path: _sat.jpg -> _mask.png
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        
        if os.path.exists(mask_path):
            tile_image_and_mask(img_path, mask_path, PROCESSED_DATA_DIR, TILE_SIZE)
        else:
            # Fallback or skip
            pass
            
    print(f"Preprocessing complete. Tiles saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
