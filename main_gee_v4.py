
"""
Sanchari V4 - GEE Inference API Server (main_gee_v4.py)

FastAPI Inference Server fetching data from Google Earth Engine.
Upgraded for V4 Architecture (EfficientNet-B4 + U-Net++, 512x512 Patching).

Usage:
    python main_gee_v4.py --debug
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

try:
    import ee
except ImportError:
    print("Error: `earthengine-api` not installed.")
    raise

# Import V4 model
# Import V4 model
from model_v4 import create_model_v4
from postprocess_v4 import apply_advanced_postprocessing

# --- Configuration ---
GEE_PROJECT = 'gen-lang-client-0330945199' 
GEE_SCALE = 1.0  
GEE_IMAGE_COLLECTION = 'USDA/NAIP/DOQQ' 

MODEL_PATH = 'weights/best_model_v4.pth'
DEBUG_DIR = 'predicted/predictedv4'
WINDOW_SIZE = 1024
PATCH_SIZE = 512 # V4
STRIDE = 256     # V4
DEBUG_MODE = False

# --- Models ---
class Coordinates(BaseModel):
    latitude: float
    longitude: float
    zoom: float = None 
    collection: str = None 

# --- State ---
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Init GEE
    try:
        ee.Initialize(project=GEE_PROJECT if GEE_PROJECT else None)
        print("GEE Initialized.")
    except Exception as e:
        print(f"GEE Init Failed: {e}")

    # 2. Load V4 Model
    print("Loading V4 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v4().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V4 weights loaded.")
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

# --- Helpers ---
def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    """Fetches GEE image crop."""
    # (Same implementation as V3, ensures correct crop for 1024x1024 input)
    point_wgs84 = ee.Geometry.Point([lon, lat])
    total_meters = size * scale
    half_span = total_meters / 2
    
    proj_3857 = ee.Projection('EPSG:3857')
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)
    
    min_x, max_x = x_center - half_span, x_center + half_span
    min_y, max_y = y_center - half_span, y_center + half_span
    
    region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y], 'EPSG:3857', False)
    
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
        print(f"GEE getThumbURL failed: {e}")
        raise HTTPException(500, f"GEE Error: {e}")
        
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(502, f"Failed to download image from GEE: {resp.status_code}")
        
    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise HTTPException(500, "Failed to decode GEE image.")
        
    if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    return image, transform, "EPSG:3857"


def predict_sliding_window(large_image, model, device):
    """V4 Inference (512x512)"""
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
            # V4 Normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            with torch.no_grad():
                probs_h = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3]).squeeze().cpu().numpy()
            with torch.no_grad():
                probs_v = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2]).squeeze().cpu().numpy()
            with torch.no_grad():
                probs_rot = torch.rot90(torch.sigmoid(model(torch.rot90(img_tensor, 1, [2, 3]))), -1, [2, 3]).squeeze().cpu().numpy()
            
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_avg
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            
    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]


@app.post("/predict")
async def predict(coords: Coordinates):
    request_id = int(time.time())
    print(f"Request: {coords.latitude}, {coords.longitude}")
    
    # 1. Fetch
    t0 = time.time()
    img_rgb, transform, crs = fetch_gee_image(
        coords.latitude, 
        coords.longitude, 
        scale=coords.zoom if coords.zoom else GEE_SCALE,
        collection=coords.collection if coords.collection else GEE_IMAGE_COLLECTION
    )
    print(f"GEE Time: {time.time() - t0:.2f}s")
    
    # 2. Inference
    prob_map = predict_sliding_window(img_rgb, model_state["model"], model_state["device"])
    
    # 3. Post-Process (Advanced V4)
    # Thresholding, Gap Closing, Skeleton Pruning
    binary_mask, skeleton = apply_advanced_postprocessing(prob_map, threshold=0.45)
    
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # 4. Debug Output
    if DEBUG_MODE:
        base_name = f"{request_id}_{coords.latitude}_{coords.longitude}"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"), img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"), (binary_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), (skeleton_uint8 * 255))
        print(f"Saved debug images to {DEBUG_DIR}")

    # 5. Vectorize
    shapes = rasterio.features.shapes(skeleton_uint8, mask=skeleton, transform=transform)
    features = []
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

    if DEBUG_MODE:
        geojson_dir = "output-geojson-v4"
        os.makedirs(geojson_dir, exist_ok=True)
        import json
        out_path = os.path.join(geojson_dir, f"{request_id}_{coords.latitude}_{coords.longitude}.geojson")
        with open(out_path, 'w') as f:
            json.dump(feature_collection, f)
        print(f"Saved debug GeoJSON to {out_path}")

    return feature_collection

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args, _ = parser.parse_known_args()
    if args.debug:
        DEBUG_MODE = True
        
    print(f"Starting GEE V4 Server. Debug: {DEBUG_MODE}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
