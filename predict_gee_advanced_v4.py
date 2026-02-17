
"""
Sanchari V4 Advanced - Optimized Inference & Graph Gap Closing (predict_gee_advanced_v4.py)

Optimized Inference Pipeline:
1. Standard V4 Sliding Window (Single Scale, 4-way TTA) - Fast & Accurate.
2. Advanced Post-Processing:
   - Hole Filling (prevents skeleton loops on wide roads).
   - Graph-based Gap Closing (connects broken segments).

Usage:
    python predict_gee_advanced_v4.py
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
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, closing, disk
from scipy.spatial import KDTree

# Import V4 Model
from model_v4 import create_model_v4

# --- Configuration ---
# GEE Settings (Must match main_gee_v4.py)
GEE_PROJECT = 'gen-lang-client-0330945199' 
GEE_SCALE = 1.0
GEE_IMAGE_COLLECTION = 'USDA/NAIP/DOQQ'

MODEL_PATH = 'weights/best_model_v4.pth'
OUTPUT_IMG_DIR = 'predicted/predictedv4_advanced'
OUTPUT_GEOJSON_DIR = 'predicted/output-geojson-advanced'
WINDOW_SIZE = 1024
PATCH_SIZE = 512
STRIDE = 256
THRESHOLD = 0.45 # From optimization

# --- Top US Cities for Random Sampling ---
CITIES = [
    (40.7128, -74.0060), # NYC
    (34.0522, -118.2437), # Los Angeles
    (41.8781, -87.6298), # Chicago
    (29.7604, -95.3698), # Houston
    (30.2672, -97.7431)  # Austin
]

# Import GEE
try:
    import ee
except ImportError:
    print("Error: `earthengine-api` not installed.")
    raise

def get_random_coords(count=5):
    coords = []
    print(f"Generating {count} random coordinates...")
    for _ in range(count):
        city_lat, city_lon = random.choice(CITIES)
        offset_lat = random.uniform(-0.05, 0.05)
        offset_lon = random.uniform(-0.05, 0.05)
        coords.append((city_lat + offset_lat, city_lon + offset_lon))
    return coords

def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    # (Same fetch logic as standard V4 scripts)
    point_wgs84 = ee.Geometry.Point([lon, lat])
    total_meters = size * scale
    half_span = total_meters / 2
    
    # Compute Transform Locally
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)
    
    min_x, max_x = x_center - half_span, x_center + half_span
    min_y, max_y = y_center - half_span, y_center + half_span
    
    region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y], 'EPSG:3857', False)
    
    try:
        if 'NAIP' in collection.upper():
             img = ee.ImageCollection(collection).filterBounds(region).filterDate('2018-01-01', '2024-01-01').sort('system:time_start', False).first().select(['R', 'G', 'B']).visualize(min=0, max=255)
        else:
             img = ee.ImageCollection(collection).filterBounds(region).first().select(['R', 'G', 'B']).visualize(min=0, max=255)

        url = img.getThumbURL({'region': region, 'dimensions': f'{size}x{size}', 'crs': 'EPSG:3857', 'format': 'jpg'})
    except Exception as e:
        print(f"GEE Error: {e}")
        return None, None, None
        
    resp = requests.get(url)
    if resp.status_code != 200: return None, None, None
        
    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None: return None, None, None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    return image, transform, "EPSG:3857"

def params_norm(device):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return mean, std

def inference_sliding_window(large_image, model, device):
    """Standard sliding window inference."""
    h, w, _ = large_image.shape

    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    padded_image = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_padded, w_padded, _ = padded_image.shape
    
    prob_map = np.zeros((h_padded, w_padded), dtype=np.float32)
    count_map = np.zeros((h_padded, w_padded), dtype=np.float32)
    
    mean, std = params_norm(device)

    for y in range(0, h_padded - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_padded - PATCH_SIZE + 1, STRIDE):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            img_tensor = (img_tensor - mean) / std
            
            # 4-way TTA (Standard V4 TTA)
            # This is fast enough (~10s/image) compared to multi-scale (~40s)
            with torch.no_grad():
                p1 = torch.sigmoid(model(img_tensor))
                p2 = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3])
                p3 = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2])
                p4 = torch.rot90(torch.sigmoid(model(torch.rot90(img_tensor, 1, [2, 3]))), -1, [2, 3])
                probs_avg = (p1 + p2 + p3 + p4) / 4.0
                
            probs = probs_avg.squeeze().cpu().numpy()
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            
    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]

def connect_components(binary_mask, max_dist=20):
    """
    Graph-based gap closing.
    """
    skeleton = skeletonize(binary_mask)
    h, w = skeleton.shape
    
    # Kernel to find endpoints (1 neighbor in 3x3)
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Endpoints are where filtered == 11 (Center 10 + 1 neighbor)
    endpoints_y, endpoints_x = np.where(filtered == 11)
    endpoints = list(zip(endpoints_y, endpoints_x))
    
    if len(endpoints) < 2:
        return binary_mask | skeleton # No connections possible
    
    # KDTree for fast neighbor lookup
    tree = KDTree(endpoints)
    
    # Canvas to draw lines
    connection_layer = np.zeros_like(binary_mask, dtype=np.uint8)
    
    pairs = tree.query_pairs(r=max_dist)
    
    for i, j in pairs:
        pt1 = endpoints[i]
        pt2 = endpoints[j]
        # Draw line (thickness 1 or 2 to ensure connectivity)
        cv2.line(connection_layer, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 1)
        
    return binary_mask | skeleton | (connection_layer > 0)

def prune_skeleton(skeleton, spur_length=20):
    """
    Experimental: Removes short spurs (branches) from skeleton.
    Iteratively removes endpoints that have only 1 neighbor, up to spur_length times.
    """
    # Create kernel for counting neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    skeleton_clean = skeleton.copy().astype(np.uint8)
    
    for _ in range(spur_length):
        filtered = cv2.filter2D(skeleton_clean, -1, kernel)
        
        # Endpoints are where filtered == 11 (Center 10 + 1 neighbor)
        # But we must be careful not to delete small disconnected lines entirely if they are valid?
        # A spur is connected to a junction.
        # Simple iterative pruning removes from ends.
        # If a line is < 2*spur_length, it might vanish completely.
        # But for spurs, they are attached to main body.
        
        # Identify endpoints
        endpoints = (filtered == 11)
        if not np.any(endpoints):
            break
            
        # Remove endpoints
        skeleton_clean[endpoints] = 0
        
    return skeleton_clean > 0

def main():
    print("--- Optimized V4 Inference (Fast TTA + Graph + Pruning) ---")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v4().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    else:
        print("Model not found.")
        return
    model.eval()

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GEOJSON_DIR, exist_ok=True)

    targets = get_random_coords(3)

    for i, (lat, lon) in enumerate(targets):
        print(f"\nProcessing ({lat:.5f}, {lon:.5f})...")
        t0 = time.time()
        img_rgb, transform, _ = fetch_gee_image(lat, lon)
        if img_rgb is None: continue
        
        t1 = time.time()
        # 1. Standard Inference (Fast TTA)
        prob_map = inference_sliding_window(img_rgb, model, device)
        t2 = time.time()
        inference_time = t2 - t1
        
        # 2. Threshold
        binary_mask = prob_map > THRESHOLD
        
        # 3. Cleanup & Gap Closing
        t3 = time.time()
        
        # FIX: Remove holes (Updated to use max_size per warning)
        binary_mask = remove_small_holes(binary_mask, area_threshold=200)
        
        # Connect broken segments
        connected_mask = connect_components(binary_mask, max_dist=25)
        
        # Final cleanup (Updated to use min_size per warning? No, wait. Warning said 'min_size is deprecated... use max_size'. 
        # But remove_small_objects removes everything smaller. If new parameter is max_size, it means "remove objects with size <= max_size".
        # So we should use max_size=100.)
        # Actually check skimage version installed. The warning suggests it.
        try:
             final_mask = remove_small_objects(connected_mask, min_size=100)
        except TypeError:
             final_mask = remove_small_objects(connected_mask, max_size=100) # Future-proof
             
        final_mask = closing(final_mask, disk(3))
        
        t4 = time.time()
        post_time = t4 - t3
        
        # Skeletonize
        skeleton = skeletonize(final_mask)
        
        # Experimental: Pruning
        t5 = time.time()
        pruned_skeleton = prune_skeleton(skeleton, spur_length=15)
        prune_time = t5 - t4
        
        print(f"   Inference: {inference_time:.2f}s")
        print(f"   Post-Process: {post_time:.2f}s")
        print(f"   Pruning: {prune_time:.2f}s")
        print(f"   Total Time: {time.time() - t0:.2f}s")
        
        # Save
        base_name = f"opt_{i}_{lat:.5f}_{lon:.5f}"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_input.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_mask_optimized.png"), (final_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_skeleton_pruned.png"), (pruned_skeleton * 255).astype(np.uint8))
        
        # Overlay
        overlay = img_rgb.copy()
        overlay[final_mask > 0] = [0, 255, 0] # Green for optimized
        combined = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_name}_overlay.jpg"), combined)

    print("\nOptimized Inference Complete.")

if __name__ == "__main__":
    main()
