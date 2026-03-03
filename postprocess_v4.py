import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, skeletonize
from typing import Tuple
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import rasterio

try:
    import sknw
except ImportError:
    raise ImportError("Please install sknw: pip install sknw")

def extract_edt_skeleton(prob_map: np.ndarray, thresh: float = 0.45, max_hole_size: int = 400) -> np.ndarray:
    """
    Step 1: Converts probability map to a robust skeleton using EDT.
    """
    # Binarize
    binary_mask = prob_map > thresh
    
    # Fill topological holes to prevent the skeleton from forming false internal cycles 
    try:
        binary_mask = remove_small_holes(binary_mask, max_size=max_hole_size)
    except TypeError:
        # Fallback for older skimage versions
        binary_mask = remove_small_holes(binary_mask, area_threshold=max_hole_size)
        
    skeleton = skeletonize(binary_mask)
    return skeleton

def build_networkx_graph(skeleton_mask: np.ndarray) -> nx.MultiGraph:
    """
    Step 2: Converts the 1-pixel Boolean skeleton mask into a NetworkX MultiGraph.
    """
    graph = sknw.build_sknw(skeleton_mask, multi=True)
    return graph

def prune_false_cycles_and_spurs(graph: nx.MultiGraph, min_spur_length: float = 20.0, max_cycle_length: float = 100.0) -> nx.MultiGraph:
    """
    Step 3: Operates topologically on the graph to prune noise and collapse false loops.
    """
    G = graph.copy()
    
    # 1. Prune short spurs (dead ends)
    spurs_removed = True
    while spurs_removed:
        spurs_removed = False
        nodes_to_remove = []
        for node in G.nodes():
            if G.degree(node) == 1:
                neighbor = list(G.neighbors(node))[0]
                edges = G[node][neighbor]
                for key, edge_data in edges.items():
                    if edge_data.get('weight', 0) < min_spur_length:
                        nodes_to_remove.append(node)
                        break 
        
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            spurs_removed = True
            
    # 2. Collapse false cycles (Small self-loops on a single node)
    edges_to_remove = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if u == v and data.get('weight', 0) < max_cycle_length:
            edges_to_remove.append((u, v, key))
            
    G.remove_edges_from(edges_to_remove)
    
    # 3. Collapse minor parallel edges (Small loops between exactly two nodes)
    parallel_edges_to_remove = []
    for u, v in G.edges():
        if u != v and G.number_of_edges(u, v) > 1:
            edges = G[u][v]
            total_perimeter = sum(data.get('weight', 0) for data in edges.values())
            if total_perimeter < max_cycle_length:
                sorted_edges = sorted(edges.items(), key=lambda item: item[1].get('weight', float('inf')))
                for key, _ in sorted_edges[1:]:
                    parallel_edges_to_remove.append((u, v, key))
                    
    G.remove_edges_from(parallel_edges_to_remove)
    
    # Clean up any isolated nodes left behind
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    return G

def process_probability_map_to_graph(prob_map: np.ndarray, thresh: float = 0.45) -> Tuple[nx.MultiGraph, np.ndarray]:
    """Wrapper function to execute Graph building."""
    skeleton = extract_edt_skeleton(prob_map, thresh=thresh)
    raw_graph = build_networkx_graph(skeleton)
    cleaned_graph = prune_false_cycles_and_spurs(raw_graph, min_spur_length=15.0, max_cycle_length=60.0)
    return cleaned_graph, skeleton

def graph_to_gdf(graph: nx.MultiGraph, transform: rasterio.Affine, crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """
    Step 4: Converts a cleaned sknw NetworkX MultiGraph into a GeoDataFrame of projected LineStrings.
    """
    lines = []
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'pts' not in data:
            continue
            
        pts = data['pts']
        
        xs = pts[:, 1]
        ys = pts[:, 0]
        
        proj_xs, proj_ys = rasterio.transform.xy(transform, ys, xs)
        
        line_coords = list(zip(proj_xs, proj_ys))
        
        if len(line_coords) >= 2:
            lines.append(LineString(line_coords))
            
    if not lines:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=crs)
        
    multiline = MultiLineString(lines)
    gdf = gpd.GeoDataFrame([{'name': 'road_network'}], geometry=[multiline], crs=crs)
    
    if crs.upper() == "EPSG:3857":
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=1.5, preserve_topology=True)
    elif crs.upper() == "EPSG:4326":
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.00001, preserve_topology=True)
        
    return gdf

def export_to_geojson(gdf: gpd.GeoDataFrame, output_path: str, target_crs: str = "EPSG:4326"):
    """
    Step 5: Exports the GeoDataFrame to a valid GeoJSON file.
    """
    if gdf.empty:
        with open(output_path, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": []}')
        return
        
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        
    gdf.to_file(output_path, driver="GeoJSON")

def apply_advanced_postprocessing(prob_map, threshold=0.45):
    """
    Applies the full V4 Advanced Post-Processing pipeline using the new Graph-Theoretic approach:
    1. Thresholding & Morphological Cleanup
    2. EDT Skeleton Extraction
    3. NetworkX Graph Building
    4. False-Cycle Collapse & Spur Pruning
    
    Returns:
        final_mask (np.ndarray): Cleaned binary mask of roads.
        skeleton (np.ndarray): The intermediate EDT skeleton mask.
        cleaned_graph (nx.MultiGraph): The topologically corrected road network.
    """
    # 1. Threshold
    binary_mask = prob_map > threshold
    
    # 2. Hole Filling (Remove small holes to prevent skeleton loops)
    try:
        binary_mask = remove_small_holes(binary_mask, max_size=400)
    except TypeError:
        binary_mask = remove_small_holes(binary_mask, area_threshold=400)

    # 3. Cleanup (Remove small noise objects)
    try:
        cleaned_mask = remove_small_objects(binary_mask, max_size=100)
    except TypeError:
        cleaned_mask = remove_small_objects(binary_mask, min_size=100)
    
    # 4. Morphological Closing (Smoothing edges before skeletonization)
    final_mask = closing(cleaned_mask, disk(3))
    
    # 5. Extract Graph (EDT Skeleton -> Raw Graph -> Cleaned Graph)
    cleaned_graph, skeleton = process_probability_map_to_graph(final_mask, thresh=0.5) 
    
    return final_mask, skeleton, cleaned_graph
