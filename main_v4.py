
"""
Sanchari V4 - API Server (main_v4.py)

FastAPI Inference Server for V4 Model (EfficientNet-B4 + U-Net++).
Process: GeoTIFF -> 1024x1024 Crop -> Sliding Window (512x512) -> GeoJSON.

Usage:
    python main_v4.py --debug
"""

import os
import glob
from contextlib import asynccontextmanager
import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import torch
import cv2
import pyproj
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from skimage.morphology import skeletonize, remove_small_objects, closing, disk
import argparse
import sys
import time

# Import V4 model
from model_v4 import create_model_v4

# --- Configuration & Constants ---
GEOTIFFS_DIR = './geotiffs/'       
MODEL_PATH = 'weights/best_model_v4.pth'   
DEBUG_DIR = 'predicted/predictedv4'          
WINDOW_SIZE = 1024                 
PATCH_SIZE = 512                   # V4: 512x512
STRIDE = 256                       # V4: 256 Overlap

# --- Global Flags ---
DEBUG_MODE = False                 

# --- Pydantic Models ---
class Coordinates(BaseModel):
    latitude: float
    longitude: float

# --- Model State ---
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading V4 model (EfficientNet-B4 + U-Net++)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model_v4().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V4 weights loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Predictions will be garbage.")
    
    model.eval()
    model_state["model"] = model
    model_state["device"] = device
    
    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug mode ENABLED. Outputting to {DEBUG_DIR}")

    yield
    model_state.clear()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# --- Helpers ---
def find_geotiff_for_coords(lon, lat):
    for tif_path in glob.glob(os.path.join(GEOTIFFS_DIR, '*.tif')):
        with rasterio.open(tif_path) as src:
            transformer = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            if src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top:
                return tif_path
    return None

def predict_sliding_window(large_image, model, device):
    """V4 Sliding Window (512x512)"""
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
            # V4 Normalization (ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            # TTA
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # H Flip
            with torch.no_grad():
                probs_h = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3]).squeeze().cpu().numpy()
            # V Flip
            with torch.no_grad():
                probs_v = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2]).squeeze().cpu().numpy()
            # Rot 90
            with torch.no_grad():
                probs_rot = torch.rot90(torch.sigmoid(model(torch.rot90(img_tensor, 1, [2, 3]))), -1, [2, 3]).squeeze().cpu().numpy()
            
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_avg
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]

# --- API ---
@app.post("/predict")
async def predict(coords: Coordinates):
    lon, lat = coords.longitude, coords.latitude
    request_id = int(time.time())
    print(f"Request: {lat}, {lon}")

    # 1. Locate Data
    geotiff_path = find_geotiff_for_coords(lon, lat)
    if not geotiff_path:
        raise HTTPException(status_code=404, detail="No GeoTIFF data found for these coordinates.")
    
    # 2. Read Window
    with rasterio.open(geotiff_path) as src:
        try:
             transformer_to_3857 = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
             if src.crs.to_epsg() != 3857:
                 transformer_to_native = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
                 target_x, target_y = transformer_to_native.transform(lon, lat)
             else:
                 target_x, target_y = transformer_to_3857.transform(lon, lat)

             row, col = src.index(target_x, target_y)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Coordinate transformation error: {e}")

        window = rasterio.windows.Window(col_off=col - WINDOW_SIZE // 2, row_off=row - WINDOW_SIZE // 2, width=WINDOW_SIZE, height=WINDOW_SIZE)
        img_array = src.read(window=window, boundless=True, fill_value=0)
        img_hwc = np.moveaxis(img_array[:3], 0, -1)
        window_transform = src.window_transform(window)

    # 3. Inference
    prob_map = predict_sliding_window(img_hwc, model_state["model"], model_state["device"])

    # 4. Post-Process (Advanced V4)
    # Thresholding, Gap Closing, Skeleton Pruning
    binary_mask, skeleton = apply_advanced_postprocessing(prob_map, threshold=0.45)
    
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # 5. Debug
    if DEBUG_MODE:
        base_name = f"{request_id}_{lat}_{lon}"
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"), img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), (skeleton_uint8 * 255))
        print(f"Saved debug images to {DEBUG_DIR}")

    # 6. Vectorize
    shapes = rasterio.features.shapes(skeleton_uint8, mask=skeleton, transform=window_transform)
    features = []
    transformer_to_wgs84 = pyproj.Transformer.from_crs(src.crs, "epsg:4326", always_xy=True)

    for geom, val in shapes:
        if val == 1:
            poly_coords = geom['coordinates'][0]
            wgs84_coords = []
            for x, y in poly_coords:
                lon_deg, lat_deg = transformer_to_wgs84.transform(x, y)
                wgs84_coords.append((lon_deg, lat_deg))
            
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": wgs84_coords},
                "properties": {} 
            })

    feature_collection = {"type": "FeatureCollection", "features": features}
    
    if DEBUG_MODE:
        geojson_dir = "output-geojson-v4"
        os.makedirs(geojson_dir, exist_ok=True)
        import json
        out_path = os.path.join(geojson_dir, f"{request_id}_{lat}_{lon}.geojson")
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
        
    print(f"Starting V4 Server. Debug: {DEBUG_MODE}")
    print("WARNING: Ensure you have trained weights/best_model_v4.pth first!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
