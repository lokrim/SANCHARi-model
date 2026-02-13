
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

# Import V3 model architecture
from model_v3 import create_model_v3

# --- Configuration & Constants ---
GEOTIFFS_DIR = './geotiffs/'       # Directory containing source GeoTIFFs
MODEL_PATH = 'weights/best_model_v3.pth'   # Path to trained model weights
DEBUG_DIR = 'predictedv3'          # Directory for debug outputs (images)
WINDOW_SIZE = 1024                 # Size of the crop taken from the GeoTIFF (Input to pipeline)
PATCH_SIZE = 256                   # Size of patches fed into the neural network (256x256)
STRIDE = 128                       # Stride for sliding window (50% overlap for better consistency)

# --- Global Flags ---
DEBUG_MODE = False                 # Toggle for saving intermediate images (controlled by --debug)

# --- Pydantic Models ---
class Coordinates(BaseModel):
    """
    Request body schema for the /predict endpoint.
    Expects WGS84 coordinates (standard Lat/Lon).
    """
    latitude: float
    longitude: float

# --- Model & Application State ---
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup (loading model) and shutdown (cleanup) logic.
    """
    print("Loading V3 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = create_model_v3().to(device)
    
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V3 weights loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Running with random weights (Predictions will be garbage).")
    
    model.eval() # Set model to evaluation mode (freezes dropout/batchnorm)
    model_state["model"] = model
    model_state["device"] = device
    
    # Setup debug directory
    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug mode ENABLED. Outputting intermediate images to {DEBUG_DIR}")

    yield # Control yields to the application here
    
    # Shutdown logic
    model_state.clear()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---

def find_geotiff_for_coords(lon, lat):
    """
    Scans the GEOTIFFS_DIR to find a file that contains the given WGS84 coordinates.
    Returns path to the matching TIFF or None.
    """
    # Note: This checks raw bounds. Since our input is WGS84, we might need
    # to transform the query point to the file's CRS if the file isn't 4326.
    # However, rasterio.open(path) usually gives bounds in native CRS.
    # Ideally, we should unify CRS. For now, assuming naive check or standard files.
    # (Improvement: We rely on check_coords.py logic for rigorous checking)
    for tif_path in glob.glob(os.path.join(GEOTIFFS_DIR, '*.tif')):
        with rasterio.open(tif_path) as src:
            # Simple bounds check (Assumes lon/lat are compatible with bounds units)
            # If TIFF is EPSG:3857, lon/lat (EPSG:4326) won't match.
            # Fix: We transform the input point to the file's CRS before checking.
            
            # Create transformer from WGS84 -> File's CRS
            transformer = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            
            if src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top:
                return tif_path
    return None

def predict_sliding_window(large_image, model, device):
    """
    Performs sliding window inference on a large image (e.g., 1024x1024).
    Uses Test Time Augmentation (TTA) and overlapping patches to improve quality.
    
    Args:
        large_image: Numpy array (H, W, 3) in BGR or RGB format.
        model: Loaded PyTorch model.
        device: 'cuda' or 'cpu'.
        
    Returns:
        prob_map: Numpy array (H, W) containing probability scores (0.0 to 1.0).
    """
    h, w, _ = large_image.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Calculate padding to ensure all edge pixels are covered by at least one patch
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    
    # Use REFLECT padding to avoid sharp edges introducing artifacts
    padded_image = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_padded, w_padded, _ = padded_image.shape
    
    # Iterate over the image with a sliding window
    for y in range(0, h_padded - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_padded - PATCH_SIZE + 1, STRIDE):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Preprocess patch: (H,W,C) -> (C,H,W), Normalize to 0-1, Add Batch Dim
            img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            
            # Normalize (match training) - V3 training uses ImageNet normalization
            # Manual normalization: (x - mean) / std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            # --- Test Time Augmentation (TTA) ---
            # 1. Original Prediction
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 2. Horizontal Flip Prediction
            img_tensor_h = torch.flip(img_tensor, [3])
            with torch.no_grad():
                logits_h = model(img_tensor_h)
                probs_h = torch.flip(torch.sigmoid(logits_h), [3]).squeeze().cpu().numpy()
            
            # 3. Vertical Flip Prediction
            img_tensor_v = torch.flip(img_tensor, [2])
            with torch.no_grad():
                logits_v = model(img_tensor_v)
                probs_v = torch.flip(torch.sigmoid(logits_v), [2]).squeeze().cpu().numpy()
            
            # 4. Rotate 90 Degrees Prediction
            img_tensor_rot = torch.rot90(img_tensor, 1, [2, 3])
            with torch.no_grad():
                logits_rot = model(img_tensor_rot)
                probs_rot = torch.rot90(torch.sigmoid(logits_rot), -1, [2, 3]).squeeze().cpu().numpy()
            
            # Average all 4 predictions for robustness
            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            
            # Accumulate predictions into the main map
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_avg
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Normalize by the number of times each pixel was predicted (averaging)
    count_map[count_map == 0] = 1 # Avoid division by zero
    prob_map /= count_map
    
    # Crop padding to return original size
    return prob_map[:h, :w]

# --- API Endpoints ---

@app.post("/predict")
async def predict(coords: Coordinates):
    """
    Main/Inference endpoint.
    1. Receives Lat/Lon.
    2. Finds matching GeoTIFF.
    3. Crops a 1024x1024 window.
    4. Runs V3 Inference (Sliding Window + TTA).
    5. Post-processes (Morphology + Skeletonization).
    6. Vectorizes result to GeoJSON.
    """
    lon, lat = coords.longitude, coords.latitude
    request_id = int(time.time())
    
    print(f"Received request: Lat={lat}, Lon={lon}")

    # 1. Locate Data
    geotiff_path = find_geotiff_for_coords(lon, lat)
    if not geotiff_path:
        raise HTTPException(status_code=404, detail="No GeoTIFF data found for these coordinates.")
    
    print(f"Found match: {geotiff_path}")

    # 2. Read Window from GeoTIFF
    with rasterio.open(geotiff_path) as src:
        # Verify Projection (Must be Web Mercator for our training assumption, or we need to reproject)
        # Note: Ideally we handle any CRS, but for now we enforce or assume compatibility.
        # Strict check:
        # if src.crs.to_epsg() != 3857:
        #     raise HTTPException(500, "GeoTIFF must be Web Mercator (EPSG:3857).")
        
        try:
             # Convert input WGS84 (Lon/Lat) -> Web Mercator (X/Y) to index into the TIFF
             transformer_to_3857 = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
             if src.crs.to_epsg() != 3857:
                 # If file is not 3857, simple transform might fail index lookup if units differ.
                 # Fallback: Transform to file's native CRS
                 transformer_to_native = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
                 target_x, target_y = transformer_to_native.transform(lon, lat)
             else:
                 target_x, target_y = transformer_to_3857.transform(lon, lat)

             row, col = src.index(target_x, target_y)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Coordinate transformation error: {e}")

        # Read 1024x1024 window centered on the point
        window = rasterio.windows.Window(col_off=col - WINDOW_SIZE // 2, row_off=row - WINDOW_SIZE // 2, width=WINDOW_SIZE, height=WINDOW_SIZE)
        img_array = src.read(window=window, boundless=True, fill_value=0)
        
        # Prepare image for Model (Stay in RGB, do NOT convert to BGR)
        img_hwc = np.moveaxis(img_array[:3], 0, -1)
        # img_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR) # REMOVED: Model expects RGB
        
        # Get transform for this specific window (vital for mapping pixels back to coords)
        window_transform = src.window_transform(window)

    # 3. Inference
    prob_map = predict_sliding_window(img_hwc, model_state["model"], model_state["device"])

    # 4. Post-Processing
    # Thresholding: 0.45 chosen based on validation optimization
    binary_mask = prob_map > 0.45 
    
    # Morphology: Clean up noise and connect gaps
    # remove_small_objects(max_size=100): Removes isolated blobs <= 100 pixels
    try:
        binary_mask = remove_small_objects(binary_mask, max_size=100)
    except TypeError:
        # Fallback for older skimage versions
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        
    # closing: Dilation followed by Erosion to bridge small gaps
    binary_mask = closing(binary_mask, footprint=disk(3))
    
    # Skeletonization: Reduce roads to 1-pixel wide centerlines
    skeleton = skeletonize(binary_mask)
    skeleton_uint8 = skeleton.astype(np.uint8)

    # 5. Debug Output (Optional)
    if DEBUG_MODE:
        base_name = f"{request_id}_{lat}_{lon}"
        # Convert RGB back to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"), img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"), (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"), (binary_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), (skeleton_uint8 * 255))
        print(f"Saved debug images to {DEBUG_DIR} for request {request_id}")

    # 6. Vectorize (GeoJSON)
    shapes = rasterio.features.shapes(skeleton_uint8, mask=skeleton, transform=window_transform)
    features = []
    
    # Transformer for resulting coordinates (Native -> WGS84 for GeoJSON output)
    transformer_to_wgs84 = pyproj.Transformer.from_crs(src.crs, "epsg:4326", always_xy=True)

    for geom, val in shapes:
        if val == 1: # If shape is foreground (road)
            poly_coords = geom['coordinates'][0]
            wgs84_coords = []
            for x, y in poly_coords:
                # Transform each point
                lon_deg, lat_deg = transformer_to_wgs84.transform(x, y)
                wgs84_coords.append((lon_deg, lat_deg))
            
            # Create GeoJSON Feature
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": wgs84_coords # GeoJSON uses [Lon, Lat]
                },
                "properties": {} 
            })

    feature_collection = {"type": "FeatureCollection", "features": features}
    
    # Save GeoJSON if Debug Mode
    if DEBUG_MODE:
        geojson_dir = "output-geojson"
        os.makedirs(geojson_dir, exist_ok=True)
        import json
        out_path = os.path.join(geojson_dir, f"{request_id}_{lat}_{lon}.geojson")
        with open(out_path, 'w') as f:
            json.dump(feature_collection, f)
        print(f"Saved debug GeoJSON to {out_path}")

    return feature_collection

if __name__ == "__main__":
    import uvicorn
    # CLI Argument Parser
    parser = argparse.ArgumentParser(description="V3 API Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (save intermediate images)")
    
    # parse_known_args allows running with uvicorn/python seamlessly
    args, unknown = parser.parse_known_args()
    
    if args.debug:
        DEBUG_MODE = True
        
    print(f"Starting V3 Server. Debug Mode: {DEBUG_MODE}")
    # example curl request on zsh
    # curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"latitude": 30.2241, "longitude": -97.7816}'
    
    # Run Server
    uvicorn.run(app, host="0.0.0.0", port=8000)
