
"""
Sanchari V4 - Local GeoTIFF API Server (main.py)

FastAPI inference server that processes locally stored GeoTIFF files.

Pipeline per request:
    1. Scan ./geotiffs/ to find the file covering the given coordinate.
    2. Crop a 1024x1024 pixel window centred on the coordinate.
    3. Run sliding-window inference (512x512, 50 % overlap) with 4-way TTA.
    4. Apply graph-theoretic post-processing (skeleton + pruning).
    5. Return a GeoJSON FeatureCollection of the road network.

Debug mode outputs intermediate images and GeoJSON to disk.

Usage:
    python main.py --debug
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import glob
import json
import time
import argparse
from contextlib import asynccontextmanager

import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import torch
import cv2
import pyproj
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import create_model
from postprocess import apply_advanced_postprocessing, graph_to_gdf, export_to_geojson


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEOTIFFS_DIR = os.path.join(BASE_DIR, "geotiffs")
MODEL_PATH   = os.path.join(BASE_DIR, "weights/best_model_v4.pth")
DEBUG_DIR    = os.path.join(BASE_DIR, "predicted/predicted")
WINDOW_SIZE  = 1024   # Size (pixels) of the image crop read from the GeoTIFF.
PATCH_SIZE   = 512    # Sliding-window patch size.
STRIDE       = 256    # Sliding-window step: 50 % overlap.

# ImageNet normalisation statistics required by the EfficientNet-B4 encoder.
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEBUG_MODE  = False
model_state = {}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    latitude:  float
    longitude: float


# ---------------------------------------------------------------------------
# Application lifespan (model loading / cleanup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the V4 model on startup and releases resources on shutdown."""
    print("Loading V4 model (EfficientNet-B4 + U-Net++) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V4 weights loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Predictions will be uninitialised.")

    model.eval()
    model_state["model"]  = model
    model_state["device"] = device

    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug mode enabled. Outputs will be written to {DEBUG_DIR}.")

    yield

    model_state.clear()
    print("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

# Allow all origins in development. Restrict `allow_origins` to a specific
# list of domains (e.g. ["https://yourdomain.com"]) before deploying to prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_geotiff_for_coords(lon, lat):
    """
    Returns the path of the first GeoTIFF in GEOTIFFS_DIR whose spatial
    extent contains the given WGS84 coordinate, or None if no match is found.

    Args:
        lon (float): Longitude in WGS84.
        lat (float): Latitude in WGS84.

    Returns:
        str or None: Path to the matching GeoTIFF.
    """
    tiff_files = glob.glob(os.path.join(GEOTIFFS_DIR, "*.tif")) + glob.glob(os.path.join(GEOTIFFS_DIR, "*.tiff"))
    for tif_path in tiff_files:
        with rasterio.open(tif_path) as src:
            transformer = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            if src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top:
                return tif_path
    return None


def predict_sliding_window(large_image, model, device):
    """
    Runs sliding-window inference with 4-way TTA on a large image.

    Args:
        large_image (np.ndarray):   RGB image (H, W, 3), uint8.
        model       (nn.Module):    Trained model in eval mode.
        device      (torch.device): Compute device.

    Returns:
        np.ndarray: Float32 probability map of shape (H, W).
    """
    h, w, _ = large_image.shape
    prob_map  = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    pad_h  = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w  = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    padded = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
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
# API endpoint
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(coords: Coordinates):
    """
    Accepts a WGS84 coordinate, locates the matching GeoTIFF, runs inference,
    and returns a GeoJSON FeatureCollection of the extracted road network.
    """
    lon, lat = coords.longitude, coords.latitude
    request_id = int(time.time())
    print(f"Request: lat={lat}, lon={lon}")

    # Step 1: Locate the GeoTIFF containing this coordinate.
    geotiff_path = find_geotiff_for_coords(lon, lat)
    if not geotiff_path:
        raise HTTPException(status_code=404, detail="No GeoTIFF data found for these coordinates.")

    # Step 2: Read a window centred on the coordinate.
    with rasterio.open(geotiff_path) as src:
        try:
            if src.crs.to_epsg() != 3857:
                transformer = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            else:
                transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            target_x, target_y = transformer.transform(lon, lat)
            row, col = src.index(target_x, target_y)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Coordinate transformation error: {e}")

        window = rasterio.windows.Window(
            col_off=col - WINDOW_SIZE // 2,
            row_off=row - WINDOW_SIZE // 2,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE,
        )
        img_array       = src.read(window=window, boundless=True, fill_value=0)
        img_hwc         = np.moveaxis(img_array[:3], 0, -1)
        window_transform = src.window_transform(window)
        src_crs_string  = src.crs.to_string()

    # Step 3: Inference.
    prob_map = predict_sliding_window(img_hwc, model_state["model"], model_state["device"])

    # Step 4: Post-processing.
    binary_mask, skeleton, cleaned_graph = apply_advanced_postprocessing(prob_map, threshold=0.45)

    mask_uint8     = (binary_mask * 255).astype(np.uint8)
    skeleton_uint8 = (skeleton    * 255).astype(np.uint8)

    # Step 5: Debug output (written only when server is started with --debug).
    if DEBUG_MODE:
        base_name = f"{request_id}_{lat}_{lon}"
        img_bgr   = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"),    img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"),     (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"),     mask_uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), skeleton_uint8)
        overlay  = img_bgr.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255]
        combined = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_overlay.jpg"), combined)
        print(f"Debug images saved to {DEBUG_DIR}.")

    # Step 6: Vectorise and return GeoJSON.
    gdf              = graph_to_gdf(cleaned_graph, window_transform, crs=src_crs_string)
    feature_collection = json.loads(gdf.to_json())

    if DEBUG_MODE:
        geojson_dir = os.path.join(BASE_DIR, "predicted/output-geojson")
        os.makedirs(geojson_dir, exist_ok=True)
        out_path = os.path.join(geojson_dir, f"{request_id}_{lat}_{lon}.geojson")
        with open(out_path, "w") as f:
            json.dump(feature_collection, f)
        print(f"Debug GeoJSON saved to {out_path}.")

    return feature_collection


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Local GeoTIFF inference API server.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output to disk.")
    args, _ = parser.parse_known_args()
    if args.debug:
        DEBUG_MODE = True

    print(f"Starting V4 local API server. Debug: {DEBUG_MODE}")
    print("Ensure weights/best_model_v4.pth exists before sending requests.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
