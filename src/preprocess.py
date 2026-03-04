
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import glob
import zipfile
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import subprocess
import argparse
import shutil


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "balraj98/deepglobe-road-extraction-dataset"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed/train")

TILE_SIZE = 512  # Patch dimensions (pixels). Larger context than V1/V2 (256).
STRIDE = 256     # Sliding-window step: TILE_SIZE // 2 gives 50 % overlap.


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset():
    """
    Downloads and extracts the DeepGlobe Road Extraction dataset from Kaggle.

    Requires the Kaggle CLI to be installed and configured with a valid
    ~/.kaggle/kaggle.json credentials file. Skips the download if the
    dataset directory already exists.
    """
    print(f"Downloading {DATASET_NAME} ...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(RAW_DATA_DIR, "train")):
        print("Dataset already present in data/raw. Skipping download.")
        return

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", RAW_DATA_DIR],
            check=True
        )

        zip_path = os.path.join(RAW_DATA_DIR, "deepglobe-road-extraction-dataset.zip")
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)

        os.remove(zip_path)
        print("Download and extraction complete.")

    except FileNotFoundError:
        print(
            "Error: 'kaggle' command not found. "
            "Install it with 'pip install kaggle' and configure ~/.kaggle/kaggle.json."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image_and_mask(image_path, mask_path, dest_dir, tile_size=512, stride=256):
    """
    Tiles a large satellite image and its binary mask into overlapping patches.

    Uses a sliding-window approach with reflect padding so that every pixel
    near the image boundary is covered by at least one complete window.
    Output tiles are written to dest_dir/images/ and dest_dir/masks/.

    Args:
        image_path (str): Path to the source satellite image (GeoTIFF or JPEG).
        mask_path  (str): Path to the corresponding binary road mask.
        dest_dir   (str): Root directory for processed output tiles.
        tile_size  (int): Side length of each square tile in pixels.
        stride     (int): Step size for the sliding window. Use tile_size // 2
                          for 50 % overlap between adjacent tiles.
    """
    base_name = os.path.basename(image_path).split(".")[0]

    # Rasterio handles GeoTIFFs robustly; convert from (Bands, H, W) to (H, W, Bands).
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.moveaxis(image, 0, -1)
        image = image[:, :, :3]  # Keep only the first three bands (RGB).

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found for {image_path}. Skipping.")
        return

    h, w, _ = image.shape

    # Pad to the nearest multiple of tile_size so the sliding window fits evenly.
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    h_padded, w_padded, _ = image.shape

    img_out_dir = os.path.join(dest_dir, "images")
    mask_out_dir = os.path.join(dest_dir, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    idx = 0
    for y in range(0, h_padded - tile_size + 1, stride):
        for x in range(0, w_padded - tile_size + 1, stride):
            img_tile = image[y : y + tile_size, x : x + tile_size]
            mask_tile = mask[y : y + tile_size, x : x + tile_size]

            out_name = f"{base_name}_{idx}"
            cv2.imwrite(
                os.path.join(img_out_dir, f"{out_name}.jpg"),
                cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(os.path.join(mask_out_dir, f"{out_name}.png"), mask_tile)
            idx += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V4 Preprocessing: tile the DeepGlobe dataset into 512x512 patches."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from Kaggle before tiling."
    )
    args = parser.parse_args()

    print("--- V4 Preprocessing Pipeline ---")
    print(f"Tile size : {TILE_SIZE}x{TILE_SIZE}")
    print(f"Stride    : {STRIDE} (50 % overlap)")

    if args.download:
        download_dataset()
    else:
        print("Skipping download (pass --download to force). Checking local data ...")

    # Remove stale processed tiles to avoid mixing versions.
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Removing existing processed data at {PROCESSED_DATA_DIR} ...")
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    search_path = os.path.join(RAW_DATA_DIR, "**", "*_sat.jpg")
    sat_files = glob.glob(search_path, recursive=True)

    if not sat_files:
        print("No satellite images found. Check data/raw structure or run with --download.")
        return

    print(f"Found {len(sat_files)} source images. Starting tiling ...")

    for img_path in tqdm(sat_files):
        # Infer the mask path by replacing the _sat suffix.
        potential_masks = [
            img_path.replace("_sat.jpg", "_mask.png"),
            img_path.replace("sat.jpg", "mask.png"),
            img_path.replace("_sat.jpg", "_mask.jpg"),
        ]
        mask_path = next((p for p in potential_masks if os.path.exists(p)), None)

        if mask_path:
            tile_image_and_mask(img_path, mask_path, PROCESSED_DATA_DIR, TILE_SIZE, STRIDE)

    print(f"Preprocessing complete. Tiles saved to {PROCESSED_DATA_DIR}.")


if __name__ == "__main__":
    main()
