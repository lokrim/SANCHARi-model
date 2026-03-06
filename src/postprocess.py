
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, skeletonize
from typing import Tuple
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import rasterio
from scipy.spatial import KDTree

try:
    import sknw
except ImportError:
    raise ImportError("Please install sknw: pip install sknw")


# ---------------------------------------------------------------------------
# Step 1 — Skeleton extraction
# ---------------------------------------------------------------------------

def extract_edt_skeleton(prob_map: np.ndarray, thresh: float = 0.45, max_hole_size: int = 400) -> np.ndarray:
    """
    Converts a probability map to a 1-pixel-wide skeleton using EDT skeletonisation.

    Fills small holes in the binary mask before skeletonising to prevent the
    skeleton from forming false internal loops inside wide road segments.

    Args:
        prob_map      (np.ndarray): Float32 probability map in [0, 1].
        thresh        (float):      Binarisation threshold.
        max_hole_size (int):        Maximum hole area (pixels) to fill.

    Returns:
        np.ndarray: Boolean skeleton mask.
    """
    binary_mask = prob_map > thresh

    try:
        binary_mask = remove_small_holes(binary_mask, max_size=max_hole_size)
    except TypeError:
        # Fallback for older scikit-image versions that use area_threshold.
        binary_mask = remove_small_holes(binary_mask, area_threshold=max_hole_size)

    skeleton = skeletonize(binary_mask)
    return skeleton


# ---------------------------------------------------------------------------
# Step 1.5 — Graph gap closing
# ---------------------------------------------------------------------------

def connect_components(binary_mask: np.ndarray, max_dist: int = 25) -> np.ndarray:
    """
    Graph-based gap closing to connect broken road segments.
    Uses skeleton endpoints and connection within max_dist.
    """
    binary_mask = binary_mask > 0
    skeleton = skeletonize(binary_mask)
    
    # Kernel to find endpoints (1 neighbor in 3x3)
    # Center pixel (10) + 1 neighbor (1) = 11
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Endpoints are where filtered == 11
    endpoints_y, endpoints_x = np.where(filtered == 11)
    endpoints = list(zip(endpoints_y, endpoints_x))
    
    if len(endpoints) < 2:
        return binary_mask | skeleton
    
    # KDTree for fast neighbor lookup
    tree = KDTree(endpoints)
    connection_layer = np.zeros_like(binary_mask, dtype=np.uint8)
    
    pairs = tree.query_pairs(r=max_dist)
    for i, j in pairs:
        pt1 = endpoints[i]
        pt2 = endpoints[j]
        cv2.line(connection_layer, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 1)
        
    return binary_mask | skeleton | (connection_layer > 0)


# ---------------------------------------------------------------------------
# Step 2 — Graph construction
# ---------------------------------------------------------------------------

def build_networkx_graph(skeleton_mask: np.ndarray) -> nx.MultiGraph:
    """
    Converts a 1-pixel Boolean skeleton into a NetworkX MultiGraph via sknw.

    Nodes represent junctions and endpoints; edges carry the pixel-coordinate
    path of each road segment and its Euclidean length as the edge weight.

    Args:
        skeleton_mask (np.ndarray): Boolean skeleton array.

    Returns:
        nx.MultiGraph: Graph representation of the skeleton.
    """
    graph = sknw.build_sknw(skeleton_mask, multi=True)
    return graph


# ---------------------------------------------------------------------------
# Step 3 — Graph pruning
# ---------------------------------------------------------------------------

def prune_false_cycles_and_spurs(
    graph: nx.MultiGraph,
    min_spur_length: float = 20.0,
    max_cycle_length: float = 100.0
) -> nx.MultiGraph:
    """
    Cleans the road graph by removing spurious structures.

    Three pruning passes are applied:
      1. Spur removal  — iteratively removes degree-1 nodes whose connecting
                         edge is shorter than min_spur_length.
      2. Self-loop collapse — removes self-loops shorter than max_cycle_length
                              (artefacts from noisy filled blobs).
      3. Parallel-edge collapse — for pairs of nodes connected by multiple
                                  edges whose combined perimeter is below
                                  max_cycle_length, only the shortest edge
                                  is retained.

    Args:
        graph           (nx.MultiGraph): Input skeleton graph.
        min_spur_length (float):         Minimum spur length to retain (pixels).
        max_cycle_length (float):        Maximum loop perimeter to collapse (pixels).

    Returns:
        nx.MultiGraph: Cleaned graph with isolated nodes removed.
    """
    G = graph.copy()

    # --- Pass 1: Spur pruning ---
    spurs_removed = True
    while spurs_removed:
        spurs_removed = False
        nodes_to_remove = []
        for node in G.nodes():
            if G.degree(node) == 1:
                neighbor = list(G.neighbors(node))[0]
                for key, edge_data in G[node][neighbor].items():
                    if edge_data.get("weight", 0) < min_spur_length:
                        nodes_to_remove.append(node)
                        break
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            spurs_removed = True

    # --- Pass 2: Self-loop collapse ---
    edges_to_remove = [
        (u, v, key)
        for u, v, key, data in G.edges(keys=True, data=True)
        if u == v and data.get("weight", 0) < max_cycle_length
    ]
    G.remove_edges_from(edges_to_remove)

    # --- Pass 3: Parallel-edge collapse ---
    parallel_edges_to_remove = []
    for u, v in list(G.edges()):
        if u != v and G.number_of_edges(u, v) > 1:
            edges = G[u][v]
            total_perimeter = sum(d.get("weight", 0) for d in edges.values())
            if total_perimeter < max_cycle_length:
                # Keep only the shortest edge; schedule all others for removal.
                sorted_edges = sorted(
                    edges.items(),
                    key=lambda item: item[1].get("weight", float("inf"))
                )
                for key, _ in sorted_edges[1:]:
                    parallel_edges_to_remove.append((u, v, key))
    G.remove_edges_from(parallel_edges_to_remove)

    # Remove isolated nodes left by the pruning passes.
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


# ---------------------------------------------------------------------------
# Internal wrapper
# ---------------------------------------------------------------------------

def process_probability_map_to_graph(
    prob_map: np.ndarray,
    thresh: float = 0.45
) -> Tuple[nx.MultiGraph, np.ndarray]:
    """
    Convenience wrapper: runs the full skeleton-to-graph pipeline.

    Args:
        prob_map (np.ndarray): Float32 probability map.
        thresh   (float):      Binarisation threshold passed to extract_edt_skeleton.

    Returns:
        tuple: (cleaned_graph, skeleton_mask)
    """
    skeleton = extract_edt_skeleton(prob_map, thresh=thresh)
    raw_graph = build_networkx_graph(skeleton)
    cleaned_graph = prune_false_cycles_and_spurs(
        raw_graph,
        min_spur_length=15.0,
        max_cycle_length=60.0
    )
    return cleaned_graph, skeleton


# ---------------------------------------------------------------------------
# Step 4 — Vectorisation
# ---------------------------------------------------------------------------

def graph_to_gdf(
    graph: nx.MultiGraph,
    transform: rasterio.transform.Affine,
    crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    Converts a cleaned sknw graph into a WGS84 GeoDataFrame of LineStrings.

    Pixel-space coordinates stored in each edge's 'pts' array are transformed
    to projected map coordinates using the provided affine transform, then
    reprojected to EPSG:4326 (WGS84 lon/lat) so the result is valid GeoJSON.

    Args:
        graph     (nx.MultiGraph):         Cleaned road network graph.
        transform (rasterio.Affine):       Affine transform mapping pixels to CRS units.
        crs       (str):                   Source CRS of the affine transform (default EPSG:3857).

    Returns:
        gpd.GeoDataFrame: WGS84 GeoDataFrame with a single MultiLineString geometry.
                          Returns an empty GeoDataFrame if no edges remain.
    """
    lines = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        if "pts" not in data:
            continue

        pts = data["pts"]
        node_u = graph.nodes[u].get("o")
        node_v = graph.nodes[v].get("o")

        if node_u is not None and node_v is not None and len(pts) > 0:
            dist_u_first = (pts[0][0] - node_u[0])**2 + (pts[0][1] - node_u[1])**2
            dist_u_last = (pts[-1][0] - node_u[0])**2 + (pts[-1][1] - node_u[1])**2
            
            if dist_u_first <= dist_u_last:
                full_pts = np.vstack([node_u, pts, node_v])
            else:
                full_pts = np.vstack([node_v, pts, node_u])
        else:
            full_pts = pts

        xs = full_pts[:, 1]
        ys = full_pts[:, 0]
        proj_xs, proj_ys = rasterio.transform.xy(transform, ys, xs)

        line_coords = list(zip(proj_xs, proj_ys))
        if len(line_coords) >= 2:
            lines.append(LineString(line_coords))

    if not lines:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    multiline = MultiLineString(lines)
    gdf = gpd.GeoDataFrame([{"name": "road_network"}], geometry=[multiline], crs=crs)

    # Simplify in the source (projected) CRS first — tolerance in metres.
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=1.5, preserve_topology=True)

    # Always reproject to WGS84 so callers receive valid GeoJSON lon/lat coordinates.
    gdf = gdf.to_crs("EPSG:4326")

    # Light simplification in degree units after reprojection (~1.5 m at equator).
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.000015, preserve_topology=True)

    return gdf


# ---------------------------------------------------------------------------
# Step 5 — GeoJSON export
# ---------------------------------------------------------------------------

def export_to_geojson(gdf: gpd.GeoDataFrame, output_path: str, target_crs: str = "EPSG:4326"):
    """
    Exports a GeoDataFrame to a GeoJSON file in the target CRS.

    Writes an empty FeatureCollection if the GeoDataFrame has no features.

    Args:
        gdf         (gpd.GeoDataFrame): Road network GeoDataFrame.
        output_path (str):              Destination file path.
        target_crs  (str):              Output CRS (default EPSG:4326 / WGS84).
    """
    if gdf.empty:
        with open(output_path, "w") as f:
            f.write('{"type": "FeatureCollection", "features": []}')
        return

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf.to_file(output_path, driver="GeoJSON")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_advanced_postprocessing(
    prob_map: np.ndarray,
    threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, nx.MultiGraph]:
    """
    Runs the complete V4 post-processing pipeline on a raw probability map.

    Pipeline stages:
        1. Threshold the probability map to a binary mask.
        2. Fill small holes to prevent false skeleton loops.
        3. Remove small noise objects (speckle).
        4. Apply morphological closing to smooth road boundaries.
        5. Extract a 1-pixel skeleton via EDT skeletonisation.
        6. Convert the skeleton to a NetworkX graph (sknw).
        7. Prune spurious spurs, self-loops, and parallel edges.

    Args:
        prob_map  (np.ndarray): Float32 probability map in [0, 1].
        threshold (float):      Binarisation threshold.

    Returns:
        tuple:
            final_mask    (np.ndarray):    Cleaned binary mask.
            skeleton      (np.ndarray):    EDT skeleton mask (Boolean).
            cleaned_graph (nx.MultiGraph): Topologically corrected road graph.
    """
    # Stage 1: Threshold.
    binary_mask = prob_map > threshold

    # Stage 2: Fill small holes to prevent false skeleton loops.
    try:
        binary_mask = remove_small_holes(binary_mask, max_size=400)
    except TypeError:
        binary_mask = remove_small_holes(binary_mask, area_threshold=400)

    # Stage 2.5: Connect Components (Graph Gap Closing)
    connected_mask = connect_components(binary_mask, max_dist=25)

    # Stage 3: Remove small noise objects.
    try:
        cleaned_mask = remove_small_objects(connected_mask, max_size=100)
    except TypeError:
        cleaned_mask = remove_small_objects(connected_mask, min_size=100)

    # Stage 4: Morphological closing to smooth edges before skeletonisation.
    final_mask = closing(cleaned_mask, disk(3))

    # Stages 5-7: Skeleton extraction and graph construction / pruning.
    cleaned_graph, skeleton = process_probability_map_to_graph(final_mask, thresh=0.5)

    return final_mask, skeleton, cleaned_graph
