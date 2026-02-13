import os
import glob
import rasterio
import argparse
import pyproj
from pyproj import Transformer
from prettytable import PrettyTable

# --- Configuration ---
GEOTIFFS_DIR = './geotiffs/' # Directory containing GeoTIFF files

def get_geotiff_info(geotiff_dir):
    """
    Scans a directory for GeoTIFF files and extracts their metadata.
    
    Args:
        geotiff_dir (str): Path to the directory containing .tif files.
        
    Returns:
        list: A list of dictionaries, each containing:
              - 'file': Filename
              - 'crs': Coordinate Reference System object
              - 'bounds': Bounding box object
              - 'path': Full file path
    """
    tif_info = []
    if not os.path.exists(geotiff_dir):
        print(f"Error: Directory '{geotiff_dir}' not found.")
        return []
        
    # Find all .tif and .tiff files
    tiff_files = glob.glob(os.path.join(geotiff_dir, '*.tif')) + glob.glob(os.path.join(geotiff_dir, '*.tiff'))
    
    for tif_path in sorted(tiff_files):
        try:
            with rasterio.open(tif_path) as src:
                info = {
                    'file': os.path.basename(tif_path),
                    'path': tif_path,
                    'crs': src.crs,
                    'bounds': src.bounds
                }
                tif_info.append(info)
        except Exception as e:
            print(f"Error reading {tif_path}: {e}")
            
    return tif_info

def get_bbox_in_wgs84(bounds, crs):
    """
    Returns the bounding box key points in WGS84 (EPSG:4326).
    
    Args:
        bounds (rasterio.coords.BoundingBox): The bounds in the source CRS.
        crs (rasterio.crs.CRS): The source CRS.
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    # Create transformer to WGS84
    transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    
    # Transform corners
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    
    return min_lon, min_lat, max_lon, max_lat

def main():
    parser = argparse.ArgumentParser(description="Check Lat/Lon (WGS84) against GeoTIFF coverage.")
    parser.add_argument("latitude", type=float, help="Latitude in WGS84 (EPSG:4326)")
    parser.add_argument("longitude", type=float, help="Longitude in WGS84 (EPSG:4326)")
    args = parser.parse_args()

    input_lat = args.latitude
    input_lon = args.longitude

    print(f"\n--- Trace Checking Tool ---")
    print(f"Input Point (WGS84): Lat={input_lat}, Lon={input_lon}")

    # Convert Input to Web Mercator (EPSG:3857) for reference
    try:
        transformer_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        input_x, input_y = transformer_to_3857.transform(input_lon, input_lat)
        print(f"Input Point (EPSG:3857): X={input_x:.2f}, Y={input_y:.2f}\n")
    except Exception as e:
        print(f"Warning: Could not convert to Web Mercator: {e}")
        input_x, input_y = 0, 0

    # Get GeoTIFF inventory
    geotiff_inventory = get_geotiff_info(GEOTIFFS_DIR)
    
    if not geotiff_inventory:
        print(f"No GeoTIFFs found or processed in {GEOTIFFS_DIR}")
        return

    print(f"--- GeoTIFF Inventory ({len(geotiff_inventory)} files) ---")
    
    found_match = False
    matching_files = []

    # Table Setup
    table = PrettyTable()
    table.field_names = ["File", "CRS", "Min Lon", "Min Lat", "Max Lon", "Max Lat", "Center (Lat, Lon)", "Contains Point?"]
    table.align = "l"

    for item in geotiff_inventory:
        filename = item['file']
        crs = item['crs']
        bounds = item['bounds']
        
        try:
            # 1. CRS Info
            crs_code = crs.to_string() if crs else "Unknown"
            
            # 2. Bounding Box in WGS84
            min_lon, min_lat, max_lon, max_lat = get_bbox_in_wgs84(bounds, crs)
            
            # 3. Calculate Center
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            center_str = f"{center_lat:.4f} {center_lon:.4f}"
            
            # 4. Check Containment
            # We check in the file's native CRS to be precise
            is_contained = False
            
            # Transform input point to file's native CRS
            t = Transformer.from_crs("epsg:4326", crs, always_xy=True)
            tx, ty = t.transform(input_lon, input_lat)
            
            if bounds.left <= tx <= bounds.right and bounds.bottom <= ty <= bounds.top:
                is_contained = True

            status = "✅ YES" if is_contained else "❌ NO"
            if is_contained:
                found_match = True
                matching_files.append(filename)

            table.add_row([
                filename, 
                crs_code, 
                f"{min_lon:.4f}", 
                f"{min_lat:.4f}", 
                f"{max_lon:.4f}", 
                f"{max_lat:.4f}",
                center_str,
                status
            ])

        except Exception as e:
            err_msg = str(e)[:20] + "..." if len(str(e)) > 20 else str(e)
            table.add_row([filename, "ERROR", "-", "-", "-", "-", "-", err_msg])

    print(table)
    print("\n--- Summary ---")
    if found_match:
        print(f"✅ Success! The point ({input_lat}, {input_lon}) is covered by:")
        for f in matching_files:
            print(f"  - {f}")
    else:
        print(f"❌ Failure. The point ({input_lat}, {input_lon}) is NOT covered by any available GeoTIFF.")

if __name__ == "__main__":
    main()
