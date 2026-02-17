"""
Sanchari V3 - Batch GEE Inference Script (predict_gee_v3.py)

This script performs batch inference on random locations using Google Earth Engine.
It is designed to test the model's generalization across different US cities.

Key Functionality:
1.  **Smart Sampling**: Generates 10 random coordinates near major US cities 
    (e.g., NYC, LA, Chicago, Austin) to ensure valid NAIP imagery and road presence.
2.  **GEE Integration**: Reuses the logic from `main_gee_v3.py` to fetch RGB crops.
3.  **Full Pipeline**: Runs Inference -> TTA -> Thresholding -> Morphology -> Skeletonization.
4.  **Outputs**:
    - `predicted/predictedv3/`: Saves Input, Probability Map, Binary Mask, Skeleton, and Overlay.
    - `predicted/output-geojson/`: Saves vector data for each prediction.

Usage:
    python predict_gee_v3.py
"""

import os
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
from skimage.morphology import skeletonize, remove_small_objects, closing, disk

# Import GEE
try:
    import ee
except ImportError:
    print("Error: `earthengine-api` not installed.")
    raise

# Import V3 model
from model_v3 import create_model_v3

# --- Configuration ---
# GEE Settings (Must match main_gee_v3.py for consistency)
from main_gee_v3 import GEE_PROJECT, GEE_SCALE, GEE_IMAGE_COLLECTION

MODEL_PATH = '../../weights/best_model_v3.pth'
OUTPUT_IMG_DIR = '../../predicted/predictedv3'
OUTPUT_GEOJSON_DIR = '../../predicted/output-geojson'
WINDOW_SIZE = 1024
PATCH_SIZE = 256
STRIDE = 128
THRESHOLD = 0.3  # As per user's latest update

# --- Top US Cities for Random Sampling ---
# We use this to ensure we hit areas with roads and valid NAIP coverage.
# Format: (Lat, Lon)
CITIES = [
    (40.7128, -74.0060), # NYC
    (34.0522, -118.2437), # Los Angeles
    (41.8781, -87.6298), # Chicago
    (29.7604, -95.3698), # Houston
    (33.4484, -112.0740), # Phoenix
    (39.9526, -75.1652), # Philadelphia
    (29.4241, -98.4936), # San Antonio
    (32.7157, -117.1611), # San Diego
    (32.7767, -96.7970), # Dallas
    (30.2672, -97.7431)  # Austin
]

def get_random_coords(count=10):
    """
    Generates random coordinates near major US cities to ensure road presence.
    
    Why this approach?
    - Randomly picking points in the US bounding box often lands in deserts, forests, 
      or oceans where roads are sparse and NAIP coverage might be missing.
    - By centering on major metros (NYC, LA, etc.) and adding a small random jitter 
      (~5km), we guarantee interesting road networks for testing.
    """
    coords = []
    print(f"Generating {count} random coordinates near major US cities...")
    for _ in range(count):
        # Pick a random city
        city_lat, city_lon = random.choice(CITIES)
        
        # Add random offset (approx +/- 5km)
        # 1 deg lat ~ 111km. 0.05 deg ~ 5.5km
        # 1 deg lon ~ 80-100km. 0.05 deg ~ 4-5km
        offset_lat = random.uniform(-0.05, 0.05)
        offset_lon = random.uniform(-0.05, 0.05)
        
        lat = city_lat + offset_lat
        lon = city_lon + offset_lon
        coords.append((lat, lon))
        
    return coords

def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    """
    Fetches image from GEE. 
    (Logic copied from main_gee_v3.py to ensure 1:1 parity without circular import issues on main block)
    """
    point_wgs84 = ee.Geometry.Point([lon, lat])
    
    total_meters = size * scale
    half_span = total_meters / 2
    
    # Compute Transform Locally
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)
    
    min_x = x_center - half_span
    max_x = x_center + half_span
    min_y = y_center - half_span
    max_y = y_center + half_span
    
    region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y], 'EPSG:3857', False)
    
    # Collection Logic (Patched)
    if 'SENTINEL' in collection.upper():
         img = ee.ImageCollection(collection).filterBounds(region).sort('CLOUDY_PIXEL_PERCENTAGE').first().select(['B4', 'B3', 'B2']).visualize(min=0, max=3000)
    elif 'NAIP' in collection.upper():
         img = ee.ImageCollection(collection).filterBounds(region).filterDate('2018-01-01', '2024-01-01').sort('system:time_start', False).first().select(['R', 'G', 'B']).visualize(min=0, max=255)
    else:
         try:
             col = ee.ImageCollection(collection).filterBounds(region)
             if col.size().getInfo() > 0:
                 img = col.sort('system:time_start', False).first().select(['R', 'G', 'B']).visualize(min=0, max=255)
             else:
                 img = ee.Image(collection).select(['R', 'G', 'B'])
         except:
             img = ee.Image(collection)

    try:
        url = img.getThumbURL({
            'region': region,
            'dimensions': f'{size}x{size}',
            'crs': 'EPSG:3857',
            'format': 'jpg'
        })
    except Exception as e:
        print(f"   [WARN] GEE Fetch failed for ({lat}, {lon}): {e}")
        return None, None, None
        
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"   [WARN] Download failed: {resp.status_code}")
        return None, None, None
        
    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        return None, None, None
        
    if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    return image, transform, "EPSG:3857"

def predict_sliding_window(large_image, model, device):
    """
    Inference with TTA (Copied from main_gee_v3.py/predict_v3.py for parity)
    """
    h, w, _ = large_image.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    padded_image = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_padded, w_padded, _ = padded_image.shape
    
    for y in range(0, h_padded - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_padded - PATCH_SIZE + 1, STRIDE):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            # TTA steps
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # H Flip
            img_tensor_h = torch.flip(img_tensor, [3])
            with torch.no_grad():
                logits_h = model(img_tensor_h)
                probs_h = torch.flip(torch.sigmoid(logits_h), [3]).squeeze().cpu().numpy()
                
            # V Flip
            img_tensor_v = torch.flip(img_tensor, [2])
            with torch.no_grad():
                logits_v = model(img_tensor_v)
                probs_v = torch.flip(torch.sigmoid(logits_v), [2]).squeeze().cpu().numpy()
            
            # Rot 90
            img_tensor_rot = torch.rot90(img_tensor, 1, [2, 3])
            with torch.no_grad():
                logits_rot = model(img_tensor_rot)
                probs_rot = torch.rot90(torch.sigmoid(logits_rot), -1, [2, 3]).squeeze().cpu().numpy()
            
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_avg
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            
    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]

def main():
    print(f"--- Predict GEE V3 (Batch) ---")
    
    # 1. Initialize GEE
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE Initialized (Project: {GEE_PROJECT})")
    except Exception as e:
        print(f"GEE Init Failed: {e}")
        return

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v3().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        return
    model.eval()

    # 3. Setup Directories
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GEOJSON_DIR, exist_ok=True)

    # 4. Generate Coordinates
    targets = get_random_coords(10)

    # 5. Process loop
    for i, (lat, lon) in enumerate(targets):
        print(f"\n[{i+1}/10] Processing ({lat:.5f}, {lon:.5f})...")
        
        # Fetch
        t0 = time.time()
        img_rgb, transform, crs = fetch_gee_image(lat, lon)
        if img_rgb is None:
            print("   Skipping due to GEE fetch error.")
            continue
        print(f"   Fetch time: {time.time() - t0:.2f}s")

        # Inference
        t1 = time.time()
        prob_map = predict_sliding_window(img_rgb, model, device)
        print(f"   Inference time: {time.time() - t1:.2f}s")
        
        # Post-Process
        binary_mask = prob_map > THRESHOLD
        try:
            binary_mask = remove_small_objects(binary_mask, max_size=100)
        except:
            binary_mask = remove_small_objects(binary_mask, min_size=100)
        binary_mask = closing(binary_mask, footprint=disk(3))
        skeleton = skeletonize(binary_mask)
        skeleton_uint8 = skeleton.astype(np.uint8)
        
        # Save Images
        base_name = f"gee_batch_{i}_{lat:.5f}_{lon:.5f}"
        
        # Input (RGB -> BGR for OpenCV)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_input.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_mask.png"), (binary_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_skeleton.png"), (skeleton_uint8 * 255))

        overlay = img_rgb.copy()
        overlay[binary_mask > 0] = [0, 0, 255]
        combined = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_overlay.jpg"), combined)
        
        # Save GeoJSON
        shapes = rasterio.features.shapes(skeleton_uint8, mask=skeleton, transform=transform)
        features = []
        transformer_to_wgs84 = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        
        for geom, val in shapes:
            if val == 1:
                poly_coords = geom['coordinates'][0]
                wgs84_line = []
                for x, y in poly_coords:
                    lon_deg, lat_deg = transformer_to_wgs84.transform(x, y)
                    wgs84_line.append((lon_deg, lat_deg))
                
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": wgs84_line},
                    "properties": {"confidence": "high"}
                })
        
        geojson_path = os.path.join(OUTPUT_GEOJSON_DIR, f"{base_name}.geojson")
        with open(geojson_path, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
            
        print(f"   Saved outputs to {OUTPUT_IMG_DIR} and {OUTPUT_GEOJSON_DIR}")

    print("\n--- Batch Prediction Complete ---")

if __name__ == "__main__":
    main()
