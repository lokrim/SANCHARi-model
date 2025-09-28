
import os
import glob
import rasterio
import argparse

# --- Configuration ---
# Directory where your GeoTIFF files are stored.
GEOTIFFS_DIR = './geotiffs/'

def find_tiff_for_web_mercator_coords(lon, lat):
    """
    Scans a directory of GeoTIFFs to find one that contains the given
    Web Mercator (EPSG:3857) coordinates.

    Args:
        lon (float): The longitude in Web Mercator projection.
        lat (float): The latitude in Web Mercator projection.

    Returns:
        str: The path to the matching GeoTIFF file, or None if no match is found.
    """
    print(f"Searching for TIFFs in: {os.path.abspath(GEOTIFFS_DIR)}")
    print(f"Checking for coordinates: Longitude={lon}, Latitude={lat}\n")

    # Find all files ending with .tif or .tiff
    search_patterns = [os.path.join(GEOTIFFS_DIR, '*.tif'), os.path.join(GEOTIFFS_DIR, '*.tiff')]
    tiff_files = []
    for pattern in search_patterns:
        tiff_files.extend(glob.glob(pattern))

    if not tiff_files:
        print(f"Error: No GeoTIFF files found in '{GEOTIFFS_DIR}'.")
        return None

    for tif_path in tiff_files:
        try:
            with rasterio.open(tif_path) as src:
                print(f"Checking file: {os.path.basename(tif_path)}...")
                
                # Ensure the CRS is Web Mercator (EPSG:3857)
                if src.crs and src.crs.to_epsg() != 3857:
                    print(f"  - Warning: CRS is {src.crs.to_string()}, not EPSG:3857. Skipping bounds check for this file.")
                    continue

                # Check if the coordinates are within the file's bounds
                if src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top:
                    print(f"\nSuccess! Coordinates FOUND in file: {os.path.basename(tif_path)}")
                    print(f"  - File Bounds (EPSG:3857): {src.bounds}")
                    return tif_path
                else:
                    print("  - Coordinates are outside this file's bounds.")

        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading {tif_path}: {e}")
    
    print("\nResult: Coordinates were NOT FOUND in any GeoTIFF in the directory.")
    return None

def main():
    """
    Main function to parse arguments and run the check.
    """
    parser = argparse.ArgumentParser(
        description="Check if Web Mercator coordinates exist within any GeoTIFF in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("longitude", type=float, help="Longitude in Web Mercator coordinates (EPSG:3857).")
    parser.add_argument("latitude", type=float, help="Latitude in Web Mercator coordinates (EPSG:3857).")

    args = parser.parse_args()

    find_tiff_for_web_mercator_coords(args.longitude, args.latitude)

if __name__ == "__main__":
    # Example Usage from the command line:
    # python check_coords.py -13626988.83 4549552.98
    main()
