
"""
SANCHARi V1 - GEE Diagnostics Utility (check_gee.py)

Note: Google Earth Engine (GEE) is NOT used in V1. This script only verifies
that GEE credentials are present and that the earthengine-api can authenticate,
which is useful if you plan to upgrade to a later version that does use GEE.

Verification steps:
    1. Credential file presence.
    2. GEE initialisation (no project required in V1).
    3. Basic data catalog access check.

Usage:
    python src/check_gee.py
"""

import ee
import os


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
# V1 does not use GEE — no project ID is required.
# If you have a GEE project configured, set it here manually.
GEE_PROJECT = None
print("    GEE is not used in V1. GEE_PROJECT set to None.")


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
