
import os
import glob
import argparse
import datetime
import json
from contextlib import asynccontextmanager

import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pyproj
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from skimage.morphology import skeletonize

# --- Argument Parsing for Debug Mode ---
parser = argparse.ArgumentParser(description="FastAPI Road Segmentation Server")
parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate files.")
args = parser.parse_args()
DEBUG_MODE = args.debug

# --- Configuration ---
GEOTIFFS_DIR = './geotiffs/'
MODEL_PATH = 'best_model.pth'
WINDOW_SIZE = 1024
DEBUG_OUTPUT_DIR = 'debug_output/'

# --- Pydantic Model for WGS84 Request Body ---
class WGS84Coordinates(BaseModel):
    latitude: float
    longitude: float

# --- PyTorch U-Net Model Definition ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3 = Down(64, 128), Down(128, 256), Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Model Loading and App State ---
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model_state["model"] = model
    model_state["device"] = device
    model_state["wgs84_to_mercator"] = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    model_state["mercator_to_wgs84"] = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    if DEBUG_MODE: os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    print("Model and transformers loaded successfully.")
    yield
    model_state.clear()
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---
def find_geotiff_for_coords(lon, lat):
    for tif_path in glob.glob(os.path.join(GEOTIFFS_DIR, '*.tif')):
        with rasterio.open(tif_path) as src:
            if src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top:
                return tif_path
    return None

def preprocess_image(image_array):
    img = np.moveaxis(image_array, 0, -1)[:, :, :3]
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

# --- API Endpoint ---
@app.post("/predict")
async def predict(coords: WGS84Coordinates):
    if DEBUG_MODE: print(f"\n--- New Request ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Correct CRS Handling: Transform WGS84 input to EPSG:3857 for internal use
    if DEBUG_MODE: print(f"Received WGS84 coords: {coords.longitude}, {coords.latitude}")
    mercator_lon, mercator_lat = model_state["wgs84_to_mercator"].transform(coords.longitude, coords.latitude)
    if DEBUG_MODE: print(f"Transformed to EPSG:3857: {mercator_lon}, {mercator_lat}")

    # 2. Find GeoTIFF and Crop
    geotiff_path = find_geotiff_for_coords(mercator_lon, mercator_lat)
    if not geotiff_path: raise HTTPException(status_code=404, detail="Coordinates are outside data coverage area.")
    if DEBUG_MODE: print(f"Found GeoTIFF: {os.path.basename(geotiff_path)}")

    with rasterio.open(geotiff_path) as src:
        row, col = src.index(mercator_lon, mercator_lat)
        window = rasterio.windows.Window(col - WINDOW_SIZE // 2, row - WINDOW_SIZE // 2, WINDOW_SIZE, WINDOW_SIZE)
        crop = src.read(window=window, boundless=True, fill_value=0)
        window_transform = src.window_transform(window)

    # 3. Run Inference
    if DEBUG_MODE: print("Running model inference...")
    image_tensor = preprocess_image(crop).unsqueeze(0).to(model_state["device"])
    with torch.no_grad():
        logits = model_state["model"](image_tensor)
        mask = (torch.sigmoid(logits) > 0.003).cpu().numpy().squeeze().astype(np.uint8)

    # 4. Improve with Skeletonization
    if DEBUG_MODE: print("Skeletonizing mask...")
    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
    shapes = rasterio.features.shapes(skeleton, mask=(skeleton > 0), transform=window_transform)

    # 5. Generate GeoJSON
    if DEBUG_MODE: print("Generating GeoJSON...")
    all_lines = []
    for geom, val in shapes:
        if val > 0:
            line_coords = geom['coordinates'][0]
            transformed_line = [model_state["mercator_to_wgs84"].transform(x, y) for x, y in line_coords]
            all_lines.append(transformed_line)

    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiLineString", "coordinates": all_lines},
        "properties": {}
    }
    feature_collection = {"type": "FeatureCollection", "features": [feature]}

    # 6. Save Debug Files if Enabled
    if DEBUG_MODE:
        print("Saving debug files...")
        # Save input crop
        crop_to_save = np.moveaxis(crop, 0, -1)[:, :, :3]
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f'input_crop_{timestamp}.jpg'), cv2.cvtColor(crop_to_save, cv2.COLOR_RGB2BGR))
        # Save output mask
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f'output_mask_{timestamp}.png'), mask * 255)
        # Save GeoJSON
        with open(os.path.join(DEBUG_OUTPUT_DIR, f'output_geojson_{timestamp}.json'), 'w') as f:
            json.dump(feature_collection, f, indent=2)

    return feature_collection

if __name__ == "__main__":
    # To run: python api.py
    # To run in debug mode: python api.py --debug
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
