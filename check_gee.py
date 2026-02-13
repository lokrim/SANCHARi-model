
import ee
import os

print("--- Google Earth Engine Diagnostics Tool ---")

# 1. Auth Check
print("\n[1] Checking Credentials...")
creds_path = os.path.expanduser("~/.config/earthengine/credentials")
if os.path.exists(creds_path):
    print(f"   [INFO] Credentials file found at: {creds_path}")
else:
    print("   [WARN] No credentials file found at standard location.")

# 2. Project Configuration
print("\n[2] Checking Project Configuration...")
try:
    from main_gee_v3 import GEE_PROJECT
    print(f"   [INFO] Imported Project ID from main_gee_v3.py: {GEE_PROJECT}")
except ImportError:
    GEE_PROJECT = None
    print("   [WARN] Could not import from main_gee_v3.py. Using default/None.")

# 3. Initialization
print("\n[3] Attempting Initialization...")
try:
    ee.Initialize(project=GEE_PROJECT)
    print(f"   [SUCCESS] Authenticated with Project: {GEE_PROJECT}")
except Exception as e:
    print(f"   [FAIL] Initialization failed: {e}")
    print("\n   Troubleshooting:")
    print("   - Run: 'earthengine authenticate --auth_mode=notebook'")
    print("   - Check if the Earth Engine API is enabled in Google Cloud Console.")
    exit(1)

# 4. Catalog Access Check
print("\n[4] Checking Data Catalog Access...")
datasets = {
    "NAIP (0.6m, US Only) [Recommended]": "USDA/NAIP/DOQQ",
    "Sentinel-2 (10m, Global)": "COPERNICUS/S2_HARMONIZED",
    "Landsat 9 (30m, Global)": "LANDSAT/LC09/C02/T1_L2",
    "Google Hybrid (Display Only - No API)": "GOOGLE/HYBRID"
}

# Test Point (Austin, TX - High chance of NAIP coverage)
pt = ee.Geometry.Point([-97.7431, 30.2672])

for name, asset_id in datasets.items():
    print(f"\n   Checking: {name} -> '{asset_id}'")
    try:
        if asset_id == "GOOGLE/HYBRID":
            # Just check info, don't try to compute
            ee.Image(asset_id).getInfo()
            print("      [INFO] Layer exists (Visualization Only).")
        else:
            # Check for actual data availability
            col = ee.ImageCollection(asset_id).filterBounds(pt).limit(1)
            count = col.size().getInfo()
            if count > 0:
                print(f"      [SUCCESS] Accessible! Found data.")
                print(f"      Bands: {col.first().bandNames().getInfo()}")
            else:
                print("      [INFO] Accessible, but no data at test location.")
    except Exception as e:
        print(f"      [FAIL] Access Denied or Invalid: {e}")

print("\n--- Diagnostic Complete ---")
