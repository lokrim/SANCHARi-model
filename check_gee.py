
"""
Sanchari V4 - GEE Diagnostics Utility (check_gee.py)

Verifies the complete Google Earth Engine setup in four sequential steps:
    1. Credential file presence.
    2. Project ID configuration (imported from src/main_gee.py).
    3. GEE initialisation.
    4. Data catalog access for NAIP, Sentinel-2, and Landsat.

Usage:
    python check_gee.py
"""

import sys
import ee
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))


print("--- Google Earth Engine Diagnostics ---")


# ---------------------------------------------------------------------------
# Step 1: Credential file check
# ---------------------------------------------------------------------------

print("\n[1] Credential file ...")
creds_path = os.path.expanduser("~/.config/earthengine/credentials")
if os.path.exists(creds_path):
    print(f"    Found: {creds_path}")
else:
    print("    Not found at standard location (~/.config/earthengine/credentials).")


# ---------------------------------------------------------------------------
# Step 2: Project configuration
# ---------------------------------------------------------------------------

print("\n[2] Project configuration ...")
try:
    from main_gee import GEE_PROJECT
    print(f"    Project ID imported from src/main_gee.py: {GEE_PROJECT}")
except ImportError:
    GEE_PROJECT = None
    print("    Could not import from src/main_gee.py. Falling back to None.")


# ---------------------------------------------------------------------------
# Step 3: Initialisation
# ---------------------------------------------------------------------------

print("\n[3] Initialising GEE ...")
try:
    ee.Initialize(project=GEE_PROJECT)
    print(f"    Authenticated. Project: {GEE_PROJECT}")
except Exception as e:
    print(f"    Initialisation failed: {e}")
    print("\n    Troubleshooting:")
    print("    - Run: earthengine authenticate")
    print("    - Ensure the Earth Engine API is enabled in Google Cloud Console.")
    exit(1)


# ---------------------------------------------------------------------------
# Step 4: Data catalog access
# ---------------------------------------------------------------------------

print("\n[4] Checking catalog access ...")

DATASETS = {
    "NAIP (0.6m, USA)":       "USDA/NAIP/DOQQ",
    "Sentinel-2 (10m, global)": "COPERNICUS/S2_HARMONIZED",
    "Landsat 9 (30m, global)":  "LANDSAT/LC09/C02/T1_L2",
    "Google Hybrid (display only)": "GOOGLE/HYBRID",
}

# Test point: Austin, TX — high probability of NAIP coverage.
TEST_POINT = ee.Geometry.Point([-97.7431, 30.2672])

for name, asset_id in DATASETS.items():
    print(f"\n    {name} -> '{asset_id}'")
    try:
        if asset_id == "GOOGLE/HYBRID":
            ee.Image(asset_id).getInfo()
            print("      Accessible (visualisation layer only; not suitable for inference).")
        else:
            col   = ee.ImageCollection(asset_id).filterBounds(TEST_POINT).limit(1)
            count = col.size().getInfo()
            if count > 0:
                print(f"      Accessible. Bands: {col.first().bandNames().getInfo()}")
            else:
                print("      Accessible, but no imagery at test location.")
    except Exception as e:
        print(f"      Access failed: {e}")


print("\n--- Diagnostics complete ---")
