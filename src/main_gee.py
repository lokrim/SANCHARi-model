
"""
Sanchari V4 - GEE Inference API Server (main_gee.py)

FastAPI inference server that fetches satellite imagery live from Google Earth
Engine (NAIP or Sentinel-2) and returns GeoJSON road network predictions.

Pipeline per request:
    1. Fetch a 1024x1024 image from GEE centred on the given coordinate.
    2. Run sliding-window inference (512x512, 50 % overlap) with 4-way TTA.
    3. Apply graph-theoretic post-processing (skeleton + pruning).
    4. Return a GeoJSON FeatureCollection of the extracted road network.

Debug mode outputs intermediate images and GeoJSON to disk.

Usage:
    python main_gee.py --debug
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import io
import json
import time
import argparse
import requests
from contextlib import asynccontextmanager

import numpy as np
import cv2
import torch
import pyproj
import rasterio
import rasterio.features
import rasterio.transform
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
GEE_SCALE            = 1.0       # Metres per pixel.
GEE_IMAGE_COLLECTION = "USDA/NAIP/DOQQ"

MODEL_PATH  = os.path.join(BASE_DIR, "weights/best_model_v4.pth")
DEBUG_DIR   = os.path.join(BASE_DIR, "predicted/predicted")
WINDOW_SIZE = 1024   # Pixel dimensions of the fetched image region.
PATCH_SIZE  = 512    # Sliding-window patch size.
STRIDE      = 256    # Sliding-window step: 50 % overlap.

# ImageNet normalisation statistics required by the EfficientNet-B4 encoder.
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEBUG_MODE  = False
model_state = {}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    latitude:   float
    longitude:  float
    zoom:       float = None   # Optional scale override (metres per pixel).
    collection: str   = None   # Optional GEE ImageCollection asset ID override.


# ---------------------------------------------------------------------------
# Application lifespan (model loading / cleanup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialises GEE and loads the V4 model on startup; clears state on shutdown."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("GEE initialised.")
    except Exception as e:
        print(f"GEE initialisation failed: {e}")

    print("Loading V4 model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = create_model().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("V4 weights loaded.")
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

def fetch_gee_image(lat, lon, scale=GEE_SCALE, size=WINDOW_SIZE, collection=GEE_IMAGE_COLLECTION):
    """
    Fetches a square satellite image crop from Google Earth Engine.

    Projects the WGS84 coordinate to EPSG:3857, builds a square bounding box
    of (size * scale) metres on each side, and downloads the most recent
    cloud-free composite from the specified collection.

    Args:
        lat        (float): Latitude in WGS84.
        lon        (float): Longitude in WGS84.
        scale      (float): Metres per pixel.
        size       (int):   Image dimension in pixels.
        collection (str):   GEE ImageCollection asset ID.

    Returns:
        tuple: (image_rgb, affine_transform, crs_string).
               image_rgb is a uint8 (H, W, 3) RGB array.

    Raises:
        HTTPException: If the GEE tile URL cannot be generated or downloaded.
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
        print(f"GEE getThumbURL failed: {e}")
        raise HTTPException(500, f"GEE error: {e}")

    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(502, f"Failed to download GEE image: HTTP {resp.status_code}")

    image_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(500, "Failed to decode GEE image.")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    affine_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)
    return image, affine_transform, "EPSG:3857"


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
    Accepts a WGS84 coordinate, fetches satellite imagery from GEE, runs
    inference, and returns a GeoJSON FeatureCollection of the road network.
    """
    request_id = int(time.time())
    print(f"Request: lat={coords.latitude}, lon={coords.longitude}")

    # Step 1: Fetch GEE imagery.
    t0 = time.time()
    img_rgb, transform, crs = fetch_gee_image(
        coords.latitude,
        coords.longitude,
        scale=coords.zoom       if coords.zoom       else GEE_SCALE,
        collection=coords.collection if coords.collection else GEE_IMAGE_COLLECTION,
    )
    print(f"GEE fetch time: {time.time() - t0:.2f}s")

    # Step 2: Inference.
    prob_map = predict_sliding_window(img_rgb, model_state["model"], model_state["device"])

    # Step 3: Post-processing.
    binary_mask, skeleton, cleaned_graph = apply_advanced_postprocessing(prob_map, threshold=0.45)

    mask_uint8     = (binary_mask * 255).astype(np.uint8)
    skeleton_uint8 = (skeleton    * 255).astype(np.uint8)

    # Step 4: Debug output (written only when server is started with --debug).
    if DEBUG_MODE:
        base_name = f"{request_id}_{coords.latitude}_{coords.longitude}"
        img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_input.jpg"),    img_bgr)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_prob.png"),     (prob_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"),     mask_uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_skeleton.png"), skeleton_uint8)
        overlay  = img_bgr.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255]
        combined = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_overlay.jpg"), combined)
        print(f"Debug images saved to {DEBUG_DIR}.")

    # Step 5: Vectorise and return GeoJSON.
    gdf                = graph_to_gdf(cleaned_graph, transform, crs="EPSG:3857")
    feature_collection = json.loads(gdf.to_json())

    if DEBUG_MODE:
        geojson_dir = os.path.join(BASE_DIR, "predicted/output-geojson")
        os.makedirs(geojson_dir, exist_ok=True)
        out_path = os.path.join(geojson_dir, f"{request_id}_{coords.latitude}_{coords.longitude}.geojson")
        with open(out_path, "w") as f:
            json.dump(feature_collection, f)
        print(f"Debug GeoJSON saved to {out_path}.")

    return feature_collection


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="GEE inference API server.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output to disk.")
    args, _ = parser.parse_known_args()
    if args.debug:
        DEBUG_MODE = True

    print(f"Starting GEE V4 API server. Debug: {DEBUG_MODE}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
