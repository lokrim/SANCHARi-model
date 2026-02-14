"""
Sanchari V3 - GEE Inference API Server (main_gee_v3.py)

This script provides a FastAPI-based inference server that fetches satellite imagery 
directly from Google Earth Engine (GEE), negating the need for local GeoTIFF storage.

Key Features:
- Integrates `earthengine-api` to fetch 1024x1024 RGB crops on demand.
- Supports NAIP (0.6m US) and Sentinel-2 (10m Global) via `fetch_gee_image`.
- Computes precise Web Mercator (EPSG:3857) to WGS84 (EPSG:4326) transforms 
  locally to ensure the output GeoJSON vectors align perfectly with the map.
- Runs the full V3 Model Pipeline:
    1. Sliding Window Inference with TTA (4-way flip/rotate).
    2. Thresholding (0.3).
    3. Morphology (Remove small objects, Closing).
    4. Skeletonization.
    5. Vectorization to GeoJSON.

Usage:
    python main_gee_v3.py --debug
    
    # Send Request
    curl -X POST "http://localhost:8001/predict" \
         -H "Content-Type: application/json" \
         -d '{"latitude": 30.2672, "longitude": -97.7431}'
"""

import os
import io
import time
import argparse
import requests
import numpy as np
import cv2
import torch
import pyproj
import rasterio
import rasterio.features
import rasterio.transform
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from skimage.morphology import skeletonize, remove_small_objects, closing, disk

# Import GEE (Requires `earthengine-api`)
try:
    import ee
except ImportError:
    print("Error: `earthengine-api` not installed. Please install it with `pip install earthengine-api`.")
    raise

# Import V3 model architecture
from model_v3 import create_model_v3

# --- Configuration & Constants ---
# GEE Constants
GEE_PROJECT = 'gen-lang-client-0330945199' # Replace with user's GEE project if needed, or rely on default auth
GEE_SCALE = 1.0  # Meters per pixel (High Resolution). Adjust based on subscription/target.
GEE_IMAGE_COLLECTION = 'USDA/NAIP/DOQQ' # NAIP is 0.6m Resolution (US Only), closest to DeepGlobe training data.
# Alternatives:
# 'COPERNICUS/S2_HARMONIZED' (Global, 10m) - Too coarse for this model?
# 'GOOGLE/HYBRID' (Global, High Res) - NOT accessible via API for computation. 
# 'COPERNICUS/S2_HARMONIZED' (Sentinel-2) is free but 10m res.
# 'USDA/NAIP/DOQQ' is US only 1m.
# User claimed "obtained the license", so they likely have access to VHR or use a specific asset.
# I will make it an env var or easy config.

MODEL_PATH = 'weights/best_model_v3.pth'
DEBUG_DIR = 'predictedv3'
WINDOW_SIZE = 1024
PATCH_SIZE = 256
STRIDE = 128
DEBUG_MODE = False

# --- Pydantic Models ---
class Coordinates(BaseModel):
    latitude: float
    longitude: float
    # Optional overrides for power users
    zoom: float = None # Map GEE scale if needed
    collection: str = None 

# --- Model & Application State ---
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize GEE
    try:
        # Tries to use existing credentials or project
        # If project is strict, use ee.Initialize(project=GEE_PROJECT)
        ee.Initialize(project=GEE_PROJECT if GEE_PROJECT else None)
        print("Google Earth Engine Initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize Google Earth Engine: {e}")
        print("Ensure you have run `earthengine authenticate` or set credentials.")
        # We don't exit here to allow debugging, but requests will fail.

    # 2. Load Model
    print("Loading V3 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v3().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V3 weights loaded.")
    else:
        print(f"WARNING: {MODEL_PATH} not found.")
    
    model.eval()
    model_state["model"] = model
    model_state["device"] = device
    
    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug Mode: Outputting to {DEBUG_DIR}")

    yield
    model_state.clear()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---

def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    """
    Fetches a static image crop from Google Earth Engine centered at the given coordinates.
    
    Args:
        lat (float): Latitude of the center point (WGS84).
        lon (float): Longitude of the center point (WGS84).
        scale (float): Scale in meters per pixel. default=1.0 (NAIP/High Res).
        size (int): Output image size in pixels (e.g., 1024x1024).
        collection (str): GEE Asset ID (e.g., 'USDA/NAIP/DOQQ').
        
    Returns:
        tuple:
            - image_array (np.array): (H, W, 3) RGB image data in RGB format.
            - transform (Affine): Rasterio Affine transform mapping pixels to EPSG:3857 coordinates.
                                  Essential for georeferencing the output vectors.
            - crs (str): The projection used ('EPSG:3857').
            
    Methodology:
        1. Converts Lat/Lon to Web Mercator (EPSG:3857) meters.
        2. Defines a square bounding box centered on the point with side length = size * scale.
        3. Requests this exact region from GEE using `getThumbURL` with `crs='EPSG:3857'`.
           This forces GEE to reproject/resample the data to our grid.
        4. Downloads and decodes the JPG response.
    """
    # 1. Coordinate Math (WGS84 -> Web Mercator)
    # GEE works best if we request the grid in the projection we want (3857).
    point_wgs84 = ee.Geometry.Point([lon, lat])
    
    # We want a square region in meters (EPSG:3857)
    # Buffer in meters. size (pixels) * scale (meters/pixel) = total meters
    total_meters = size * scale
    half_span = total_meters / 2
    
    # Use GEE to create the buffer in 3857 directly?
    # Actually, GEE's buffer() function works in meters on the sphere (approx).
    # Safer to project the point, compute bounds, then make rectangle.
    
    proj_3857 = ee.Projection('EPSG:3857')
    
    # 2. Compute Transform Locally for precise vectorization later
    #    (Don't rely on GEE's internal buffer approximation logic for final alignment)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)
    
    min_x = x_center - half_span
    max_x = x_center + half_span
    min_y = y_center - half_span
    max_y = y_center + half_span
    
    # 3. Create Region
    # Note: GEE Geometry constructor takes [min_x, min_y, max_x, max_y] if providing a box?
    # No, usually coordinates list.
    # ee.Geometry.Rectangle([w, s, e, n], proj, geodesic)
    region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y], 'EPSG:3857', False)
    
    # 4. Filter Image
    # Select RGB bands. Name depends on collection. 
    # For 'GOOGLE/HYBRID', it's usually visualized directly.
    # If generic satellite, we can use a visualization method.
    
    # Logic to handle different collections roughly
    # Logic to handle different collections roughly
    if 'SENTINEL' in collection.upper():
         # Sentinel-2 example
         img = ee.ImageCollection(collection) \
                .filterBounds(region) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') \
                .first() \
                .select(['B4', 'B3', 'B2']) \
                .visualize(min=0, max=3000)
    elif 'NAIP' in collection.upper():
         # USDA/NAIP/DOQQ
         # It's an ImageCollection. We need to filter and mosaic.
         img = ee.ImageCollection(collection) \
                .filterBounds(region) \
                .filterDate('2018-01-01', '2024-01-01') \
                .sort('system:time_start', False) \
                .first() \
                .select(['R', 'G', 'B']) \
                .visualize(min=0, max=255)
    else:
         # Default/High Res
         try:
             # Try to treat as Image Collection first (safer for most modern assets)
             col = ee.ImageCollection(collection).filterBounds(region)
             # If it has images, take the latest
             if col.size().getInfo() > 0:
                 img = col.sort('system:time_start', False).first().select(['R', 'G', 'B']).visualize(min=0, max=255)
             else:
                 # Fallback to single Image or visual
                 img = ee.Image(collection).select(['R', 'G', 'B'])
         except:
             # Last resort (e.g. GOOGLE/HYBRID visualization)
             img = ee.Image(collection)

    # 5. Get URL
    # crs='EPSG:3857' ensures output matches our computed bounds
    # dimensions=size ensures exact pixel count
    try:
        url = img.getThumbURL({
            'region': region,
            'dimensions': f'{size}x{size}',
            'crs': 'EPSG:3857',
            'format': 'jpg'
        })
    except Exception as e:
        print(f"GEE getThumbURL failed: {e}")
        # Could be "User memory limit" or "Band selection" error.
        raise HTTPException(500, f"GEE Error: {e}")
        
    # 6. Download
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(502, f"Failed to download image from GEE: {resp.status_code}")
        
    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise HTTPException(500, "Failed to decode GEE image.")
        
    # Standardize to RGB (OpenCV is BGR)
    # Check shape
    if len(image.shape) == 2: # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else: # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # 7. Construct Affine Transform
    # Rasterio Transform: from_bounds(west, south, east, north, width, height)
    # Note: Rasterio expects bounds.
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    
    return image, transform, "EPSG:3857"


def predict_sliding_window(large_image, model, device):
    """
    Performs sliding window inference. Same as main_v3.py.
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
            
            # TTA
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


@app.post("/predict")
async def predict(coords: Coordinates):
    """
    GEE Inference Endpoint.
    1. Fetches RGB crop from GEE (1024x1024).
    2. Runs Inference.
    3. Returns GeoJSON (WGS84).
    """
    request_id = int(time.time())
    print(f"Request: {coords.latitude}, {coords.longitude}")
    
    # 1. Fetch from GEE
    t0 = time.time()
    img_rgb, transform, crs = fetch_gee_image(
        coords.latitude, 
        coords.longitude, 
        scale=coords.zoom if coords.zoom else GEE_SCALE,
        collection=coords.collection if coords.collection else GEE_IMAGE_COLLECTION
    )
    print(f"GEE Fetch took {time.time() - t0:.2f}s")
    
    # 2. Inference
    prob_map = predict_sliding_window(img_rgb, model_state["model"], model_state["device"])
    
    # 3. Post-Process
    binary_mask = prob_map > 0.3
    try:
        binary_mask = remove_small_objects(binary_mask, max_size=100)
    except:
        binary_mask = remove_small_objects(binary_mask, min_size=100)
    binary_mask = closing(binary_mask, footprint=disk(3))
    skeleton = skeletonize(binary_mask)
    skeleton_uint8 = skeleton.astype(np.uint8)
    
    # 4. Debug Output (Parity with main_v3.py)
    if DEBUG_MODE:
        base_name = f"{request_id}_{coords.latitude}_{coords.longitude}"
        
        # Save Images
        # Convert RGB back to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"), img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"), (binary_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), (skeleton_uint8 * 255))
        print(f"Saved debug images to {DEBUG_DIR} for request {request_id}")

    # 5. Vectorize
    # Use the precise local transform we calculated in fetch_gee_image
    shapes = rasterio.features.shapes(skeleton_uint8, mask=skeleton, transform=transform)
    
    features = []
    # Transform EPSG:3857 -> EPSG:4326 for output
    transformer_to_wgs84 = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    
    for geom, val in shapes:
        if val == 1:
            poly_coords = geom['coordinates'][0]
            wgs84_line = []
            for x, y in poly_coords:
                lon, lat = transformer_to_wgs84.transform(x, y)
                wgs84_line.append((lon, lat))
            
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": wgs84_line},
                "properties": {}
            })
            
    feature_collection = {"type": "FeatureCollection", "features": features}

    # Save GeoJSON if Debug Mode
    if DEBUG_MODE:
        geojson_dir = "output-geojson"
        os.makedirs(geojson_dir, exist_ok=True)
        import json
        out_path = os.path.join(geojson_dir, f"{request_id}_{coords.latitude}_{coords.longitude}.geojson")
        with open(out_path, 'w') as f:
            json.dump(feature_collection, f)
        print(f"Saved debug GeoJSON to {out_path}")

    return feature_collection

if __name__ == "__main__":
    import uvicorn
    # Same CLI as main_v3
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args, _ = parser.parse_known_args()
    if args.debug:
        DEBUG_MODE = True
        
    print(f"Starting GEE V3 Server. Debug: {DEBUG_MODE}")
    uvicorn.run(app, host="0.0.0.0", port=8001) # Port 8001 to distinguish

    # example curl request on zsh
    # curl -X POST "http://localhost:8001/predict" -H "Content-Type: application/json" -d '{"latitude": 40.76331358683374, "longitude": -73.97086746211978}'
    