
"""
Sanchari V4 - GeoTIFF Coverage Checker (check_coords.py)

Verifies whether a given WGS84 coordinate is covered by any GeoTIFF file
in the ./geotiffs/ directory.  Prints a summary table of all files found
and whether each one contains the query point.

Usage:
    python check_coords.py <latitude> <longitude>

Example:
    python check_coords.py -43.5609 172.7358
"""

import os
import glob
import argparse

import rasterio
from pyproj import Transformer
from prettytable import PrettyTable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEOTIFFS_DIR = "./geotiffs/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_geotiff_info(geotiff_dir):
    """
    Scans a directory for GeoTIFF files and returns their metadata.

    Args:
        geotiff_dir (str): Directory to search for .tif / .tiff files.

    Returns:
        list[dict]: Each dict contains: 'file', 'path', 'crs', 'bounds'.
                    Returns an empty list if the directory does not exist.
    """
    if not os.path.exists(geotiff_dir):
        print(f"Error: Directory '{geotiff_dir}' not found.")
        return []

    tiff_files = (
        glob.glob(os.path.join(geotiff_dir, "*.tif")) +
        glob.glob(os.path.join(geotiff_dir, "*.tiff"))
    )

    tif_info = []
    for tif_path in sorted(tiff_files):
        try:
            with rasterio.open(tif_path) as src:
                tif_info.append({
                    "file":   os.path.basename(tif_path),
                    "path":   tif_path,
                    "crs":    src.crs,
                    "bounds": src.bounds,
                })
        except Exception as e:
            print(f"Error reading {tif_path}: {e}")

    return tif_info


def get_bbox_in_wgs84(bounds, crs):
    """
    Reprojects the bounding box of a raster to WGS84 (EPSG:4326).

    Args:
        bounds (rasterio.coords.BoundingBox): Bounding box in the source CRS.
        crs    (rasterio.crs.CRS):            Source coordinate reference system.

    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat) in decimal degrees.
    """
    transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds.left,  bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    return min_lon, min_lat, max_lon, max_lat


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check whether a WGS84 coordinate is covered by any local GeoTIFF."
    )
    parser.add_argument("latitude",  type=float, help="Latitude in WGS84 (EPSG:4326).")
    parser.add_argument("longitude", type=float, help="Longitude in WGS84 (EPSG:4326).")
    args = parser.parse_args()

    input_lat = args.latitude
    input_lon = args.longitude

    print(f"\n--- GeoTIFF Coverage Check ---")
    print(f"Query point (WGS84): lat={input_lat}, lon={input_lon}")

    try:
        t3857 = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        input_x, input_y = t3857.transform(input_lon, input_lat)
        print(f"Query point (EPSG:3857): X={input_x:.2f}, Y={input_y:.2f}\n")
    except Exception as e:
        print(f"Warning: Could not project to EPSG:3857: {e}")

    inventory = get_geotiff_info(GEOTIFFS_DIR)
    if not inventory:
        print(f"No GeoTIFFs found in {GEOTIFFS_DIR}.")
        return

    print(f"--- GeoTIFF Inventory ({len(inventory)} files) ---")

    table = PrettyTable()
    table.field_names = [
        "File", "CRS", "Min Lon", "Min Lat", "Max Lon", "Max Lat", "Centre (Lat, Lon)", "Contains Point"
    ]
    table.align = "l"

    matching_files = []

    for item in inventory:
        filename = item["file"]
        crs      = item["crs"]
        bounds   = item["bounds"]

        try:
            crs_code = crs.to_string() if crs else "Unknown"
            min_lon, min_lat, max_lon, max_lat = get_bbox_in_wgs84(bounds, crs)
            centre_str = f"{(min_lat + max_lat) / 2:.4f}, {(min_lon + max_lon) / 2:.4f}"

            # Test containment in the file's native CRS for spatial accuracy.
            t_native = Transformer.from_crs("epsg:4326", crs, always_xy=True)
            tx, ty   = t_native.transform(input_lon, input_lat)
            contained = bounds.left <= tx <= bounds.right and bounds.bottom <= ty <= bounds.top

            if contained:
                matching_files.append(filename)

            table.add_row([
                filename,
                crs_code,
                f"{min_lon:.4f}",
                f"{min_lat:.4f}",
                f"{max_lon:.4f}",
                f"{max_lat:.4f}",
                centre_str,
                "YES" if contained else "NO",
            ])

        except Exception as e:
            err_msg = (str(e)[:20] + "...") if len(str(e)) > 20 else str(e)
            table.add_row([filename, "ERROR", "-", "-", "-", "-", "-", err_msg])

    print(table)
    print("\n--- Summary ---")
    if matching_files:
        print(f"Point ({input_lat}, {input_lon}) is covered by:")
        for f in matching_files:
            print(f"  {f}")
    else:
        print(f"Point ({input_lat}, {input_lon}) is NOT covered by any available GeoTIFF.")


if __name__ == "__main__":
    main()
