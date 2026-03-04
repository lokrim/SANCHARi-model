
"""
Sanchari V4 - GEE Batch Inference Script (predict_gee.py)

Fetches satellite imagery (NAIP or Sentinel-2) from Google Earth Engine for
a set of randomly sampled US city coordinates and runs the full V4 inference
and post-processing pipeline on each location.

Output files per location (saved to OUTPUT_IMG_DIR):
    {name}_input.jpg    -- Fetched RGB satellite image.
    {name}_prob.png     -- Raw probability map (8-bit greyscale).
    {name}_mask.png     -- Binary road mask after post-processing.
    {name}_skeleton.png -- 1-pixel-wide road centreline skeleton.
    {name}_overlay.jpg  -- Satellite image with mask overlaid in red.
    {name}.geojson      -- Road network GeoJSON (saved to OUTPUT_GEOJSON_DIR).

Usage:
    python predict_gee.py
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time
import random
import json
import requests
import numpy as np
import cv2
import torch
import pyproj
import rasterio
import rasterio.features
import rasterio.transform
from skimage.morphology import closing, disk

try:
    import ee
except ImportError:
    raise ImportError("`earthengine-api` is not installed. Run: pip install earthengine-api")

from model import create_model
from postprocess import apply_advanced_postprocessing, graph_to_gdf, export_to_geojson


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEE_PROJECT          = "gen-lang-client-0330945199"
GEE_SCALE            = 1.0       # Metres per pixel for fetched imagery.
GEE_IMAGE_COLLECTION = "USDA/NAIP/DOQQ"

MODEL_PATH           = os.path.join(BASE_DIR, "weights/best_model_v4.pth")
OUTPUT_IMG_DIR       = os.path.join(BASE_DIR, "predicted/predicted")
OUTPUT_GEOJSON_DIR   = os.path.join(BASE_DIR, "predicted/output-geojson")

WINDOW_SIZE = 1024   # Pixel dimensions of the fetched image region.
PATCH_SIZE  = 512    # Sliding-window patch size.
STRIDE      = 256    # Sliding-window step: 50 % overlap.
THRESHOLD   = 0.45   # Binarisation threshold for post-processing.

# ImageNet normalisation statistics required by the EfficientNet-B4 encoder.
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Representative US city centres used for random coordinate sampling.
CITIES = [
    (40.7128,  -74.0060),   # New York City
    (34.0522, -118.2437),   # Los Angeles
    (41.8781,  -87.6298),   # Chicago
    (29.7604,  -95.3698),   # Houston
    (33.4484, -112.0740),   # Phoenix
    (39.9526,  -75.1652),   # Philadelphia
    (29.4241,  -98.4936),   # San Antonio
    (32.7157, -117.1611),   # San Diego
    (32.7767,  -96.7970),   # Dallas
    (30.2672,  -97.7431),   # Austin
]


# ---------------------------------------------------------------------------
# Coordinate generation
# ---------------------------------------------------------------------------

def get_random_coords(count=10):
    """
    Generates random coordinates offset from major US city centres.

    Each coordinate is randomly displaced up to ±0.05 degrees in both
    latitude and longitude from a randomly selected city centre.

    Args:
        count (int): Number of coordinates to generate.

    Returns:
        list[tuple]: List of (latitude, longitude) pairs.
    """
    coords = []
    print(f"Generating {count} random coordinates near major US cities ...")
    for _ in range(count):
        city_lat, city_lon = random.choice(CITIES)
        coords.append((
            city_lat + random.uniform(-0.05, 0.05),
            city_lon + random.uniform(-0.05, 0.05),
        ))
    return coords


# ---------------------------------------------------------------------------
# GEE image fetch
# ---------------------------------------------------------------------------

def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    """
    Fetches a square satellite image crop from Google Earth Engine.

    Projects the input WGS84 coordinate to EPSG:3857, constructs a
    bounding box of (size * scale) metres on each side, and retrieves
    the most recent cloud-free image from the specified collection.

    Args:
        lat        (float): Latitude in WGS84.
        lon        (float): Longitude in WGS84.
        scale      (float): Metres per pixel defining the spatial resolution.
        size       (int):   Output image dimension in pixels (width == height).
        collection (str):   GEE ImageCollection asset ID.

    Returns:
        tuple: (image_rgb, affine_transform, crs_string) or (None, None, None)
               on failure.  image_rgb is a uint8 (H, W, 3) RGB array.
    """
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)

    half_span = (size * scale) / 2
    min_x, max_x = x_center - half_span, x_center + half_span
    min_y, max_y = y_center - half_span, y_center + half_span

    region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y], "EPSG:3857", False)

    if "SENTINEL" in collection.upper():
        img = (
            ee.ImageCollection(collection)
            .filterBounds(region)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .first()
            .select(["B4", "B3", "B2"])
            .visualize(min=0, max=3000)
        )
    elif "NAIP" in collection.upper():
        img = (
            ee.ImageCollection(collection)
            .filterBounds(region)
            .filterDate("2018-01-01", "2024-01-01")
            .sort("system:time_start", False)
            .first()
            .select(["R", "G", "B"])
            .visualize(min=0, max=255)
        )
    else:
        try:
            col = ee.ImageCollection(collection).filterBounds(region)
            img = (
                col.sort("system:time_start", False).first().select(["R", "G", "B"])
                if col.size().getInfo() > 0
                else ee.Image(collection).select(["R", "G", "B"])
            )
        except Exception:
            img = ee.Image(collection)

    try:
        url = img.getThumbURL({
            "region":     region,
            "dimensions": f"{size}x{size}",
            "crs":        "EPSG:3857",
            "format":     "jpg",
        })
    except Exception as e:
        print(f"   [WARN] GEE fetch failed for ({lat}, {lon}): {e}")
        return None, None, None

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"   [WARN] Image download failed: HTTP {resp.status_code}")
        return None, None, None

    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None, None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    affine_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    return image, affine_transform, "EPSG:3857"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_sliding_window(large_image, model, device):
    """
    Runs sliding-window inference with 4-way TTA on a large image.

    Args:
        large_image (np.ndarray):   Input RGB image (H, W, 3), uint8.
        model       (nn.Module):    Trained model in eval mode.
        device      (torch.device): Compute device.

    Returns:
        np.ndarray: Float32 probability map of shape (H, W).
    """
    h, w, _ = large_image.shape
    prob_map  = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    import math
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    num_windows_y = math.ceil((h + pad_h - PATCH_SIZE) / STRIDE) + 1 if (h + pad_h) > PATCH_SIZE else 1
    num_windows_x = math.ceil((w + pad_w - PATCH_SIZE) / STRIDE) + 1 if (w + pad_w) > PATCH_SIZE else 1

    target_h = max(PATCH_SIZE, (num_windows_y - 1) * STRIDE + PATCH_SIZE)
    target_w = max(PATCH_SIZE, (num_windows_x - 1) * STRIDE + PATCH_SIZE)

    padded = cv2.copyMakeBorder(large_image, 0, target_h - h, 0, target_w - w, cv2.BORDER_REFLECT)
    h_pad, w_pad, _ = padded.shape

    mean = NORM_MEAN.to(device)
    std  = NORM_STD.to(device)

    for y in range(0, h_pad - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_pad - PATCH_SIZE + 1, STRIDE):
            patch = padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            t = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            t = (t - mean) / std

            with torch.no_grad():
                probs     = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                probs_h   = torch.flip(torch.sigmoid(model(torch.flip(t, [3]))), [3]).squeeze().cpu().numpy()
                probs_v   = torch.flip(torch.sigmoid(model(torch.flip(t, [2]))), [2]).squeeze().cpu().numpy()
                probs_rot = torch.rot90(torch.sigmoid(model(torch.rot90(t, 1, [2, 3]))), -1, [2, 3]).squeeze().cpu().numpy()

            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            prob_map [y : y + PATCH_SIZE, x : x + PATCH_SIZE] += probs_avg
            count_map[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("--- Predict GEE V4 (Batch) ---")

    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialised (project: {GEE_PROJECT})")
    except Exception as e:
        print(f"GEE initialisation failed: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"V4 weights loaded from {MODEL_PATH}.")
    else:
        print(f"Model not found at {MODEL_PATH}.")
        return
    model.eval()

    os.makedirs(OUTPUT_IMG_DIR,     exist_ok=True)
    os.makedirs(OUTPUT_GEOJSON_DIR, exist_ok=True)

    targets = get_random_coords(10)

    for i, (lat, lon) in enumerate(targets):
        print(f"\n[{i + 1}/10] Processing ({lat:.5f}, {lon:.5f}) ...")

        t0 = time.time()
        img_rgb, transform, crs = fetch_gee_image(lat, lon)
        if img_rgb is None:
            print("   Skipping: GEE fetch error.")
            continue
        print(f"   Fetch time    : {time.time() - t0:.2f}s")

        t1 = time.time()
        prob_map = predict_sliding_window(img_rgb, model, device)
        print(f"   Inference time: {time.time() - t1:.2f}s")

        binary_mask, skeleton, cleaned_graph = apply_advanced_postprocessing(prob_map, threshold=THRESHOLD)

        mask_uint8     = (binary_mask * 255).astype(np.uint8)
        skeleton_uint8 = (skeleton    * 255).astype(np.uint8)

        base_name  = f"gee_batch_{i}_{lat:.5f}_{lon:.5f}"
        img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_input.jpg"),    img_bgr)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_prob.png"),     (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_mask.png"),     mask_uint8)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_skeleton.png"), skeleton_uint8)

        overlay  = img_bgr.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255]
        combined = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_overlay.jpg"), combined)

        gdf = graph_to_gdf(cleaned_graph, transform, crs="EPSG:3857")
        export_to_geojson(gdf, os.path.join(OUTPUT_GEOJSON_DIR, f"{base_name}.geojson"), target_crs="EPSG:4326")

        print(f"   Outputs saved to {OUTPUT_IMG_DIR} and {OUTPUT_GEOJSON_DIR}.")

    print("\n--- Batch V4 prediction complete ---")


if __name__ == "__main__":
    main()
