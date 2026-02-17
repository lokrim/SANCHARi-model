import os
import glob
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

# --- Configuration ---
GEOTIFFS_DIR = '../../geotiffs/'  # Directory where GeoTIFF files are stored
MODEL_PATH = '../../weights/best_model_v1.pth'   # Path to the pre-trained model weights
WINDOW_SIZE = 1024              # Size of the image window to crop (1024x1024 pixels)

# --- Pydantic Model for Request Body ---
class Coordinates(BaseModel):
    latitude: float
    longitude: float

# --- PyTorch U-Net Model Definition ---
# Note: This should be the same architecture as the model used for training.
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
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
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# --- Model and App State ---
# A dictionary to hold the model and other state during the app's lifespan
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the PyTorch model at startup
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Please ensure the file exists.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model_state["model"] = model
    model_state["device"] = device
    print("Model loaded successfully.")
    yield
    # Clean up the model and state at shutdown
    model_state.clear()
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---
def find_geotiff_for_coords(lon, lat):
    """Finds the first GeoTIFF in the directory that contains the given coordinates."""
    for tif_path in glob.glob(os.path.join(GEOTIFFS_DIR, '*.tif')):
        with rasterio.open(tif_path) as src:
            if src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top:
                return tif_path
    return None

def preprocess_image(image_array):
    """Prepares the image array for the PyTorch model."""
    # Assuming image_array is (bands, height, width) and we need (height, width, bands) for cv2
    img = np.moveaxis(image_array, 0, -1)
    # Keep only the first 3 bands (RGB)
    img = img[:, :, :3]
    # Normalize and convert to tensor
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Not strictly needed but good practice
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1) # HWC to CHW
    return img_tensor

# --- API Endpoint ---
@app.post("/predict")
async def predict(coords: Coordinates):
    """Predicts road segmentation for a given Web Mercator coordinate."""
    lon, lat = coords.longitude, coords.latitude

    # 1. Find the correct GeoTIFF
    geotiff_path = find_geotiff_for_coords(lon, lat)
    if not geotiff_path:
        raise HTTPException(status_code=404, detail="Coordinates are outside the data coverage area.")

    with rasterio.open(geotiff_path) as src:
        # Check if the source CRS is Web Mercator
        if src.crs.to_epsg() != 3857:
            raise HTTPException(status_code=500, detail=f"GeoTIFF {os.path.basename(geotiff_path)} is not in Web Mercator (EPSG:3857).")

        # 2. Crop Image Window
        try:
            row, col = src.index(lon, lat)
        except IndexError:
             raise HTTPException(status_code=400, detail="Coordinates could not be indexed in the source GeoTIFF.")

        window = rasterio.windows.Window(
            col_off=col - WINDOW_SIZE // 2,
            row_off=row - WINDOW_SIZE // 2,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE
        )
        crop = src.read(window=window, boundless=True, fill_value=0)
        window_transform = src.window_transform(window)

    # 3. Run PyTorch Model Inference
    image_tensor = preprocess_image(crop).unsqueeze(0).to(model_state["device"])
    with torch.no_grad():
        logits = model_state["model"](image_tensor)
        mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze().astype(np.uint8)

    # 4. Vectorize the Mask
    shapes = rasterio.features.shapes(mask, mask=mask.astype(bool), transform=window_transform)

    # 5. Generate GeoJSON with MultiLineStrings in WGS84
    transformer = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    features = []
    for geom, val in shapes:
        if val == 1: # Corresponds to road pixels
            # Transform polygon coordinates to LineString boundaries in WGS84
            lines = []
            for line_coords in geom['coordinates']:
                transformed_line = [transformer.transform(x, y) for x, y in line_coords]
                lines.append(transformed_line)
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [lines] # Wrap lines in another list for MultiLineString
                },
                "properties": {}
            })

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    return feature_collection

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    # Ensure you have a `geotiffs` folder with .tif files and `best_model.pth` in the root.
    uvicorn.run(app, host="0.0.0.0", port=8000)