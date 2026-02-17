
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, closing, disk
from scipy.spatial import KDTree

def connect_components(binary_mask, max_dist=25):
    """
    Graph-based gap closing to connect broken road segments.
    Uses skeleton endpoints and connection within max_dist.
    """
    # Ensure binary
    binary_mask = binary_mask > 0
    
    skeleton = skeletonize(binary_mask)
    h, w = skeleton.shape
    
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
    
    # Canvas to draw lines
    connection_layer = np.zeros_like(binary_mask, dtype=np.uint8)
    
    # Find pairs within max_dist
    pairs = tree.query_pairs(r=max_dist)
    
    for i, j in pairs:
        pt1 = endpoints[i]
        pt2 = endpoints[j]
        # Draw line (thickness 1 to match skeleton)
        cv2.line(connection_layer, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 1)
        
    return binary_mask | skeleton | (connection_layer > 0)

def prune_skeleton(skeleton, spur_length=15):
    """
    Removes short spurs (branches) from skeleton.
    Iteratively removes endpoints that have only 1 neighbor.
    """
    # Kernel for counting neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    skeleton_clean = skeleton.copy().astype(np.uint8)
    
    # Iterative pruning
    for _ in range(spur_length):
        filtered = cv2.filter2D(skeleton_clean, -1, kernel)
        
        # Identify endpoints (value 11)
        endpoints = (filtered == 11)
        
        if not np.any(endpoints):
            break
            
        # Remove endpoints
        skeleton_clean[endpoints] = 0
        
    return skeleton_clean > 0

def apply_advanced_postprocessing(prob_map, threshold=0.45):
    """
    Applies the full V4 Advanced Post-Processing pipeline:
    1. Thresholding
    2. Hole Filling (prevent loops)
    3. Graph Gap Closing
    4. Small Object Removal
    5. Morphological Closing
    6. Skeletonization
    7. Skeleton Pruning
    
    Returns:
        final_mask (np.array): Binary mask of roads.
        final_skeleton (np.array): Pruned centerline skeleton.
    """
    # 1. Threshold
    binary_mask = prob_map > threshold
    
    # 2. Hole Filling (Remove small holes to prevent skeleton loops)
    # Use max_size (new skimage) or area_threshold (old)
    try:
        binary_mask = remove_small_holes(binary_mask, max_size=200)
    except TypeError:
        binary_mask = remove_small_holes(binary_mask, area_threshold=200)

    # 3. Connect Components (Graph Gap Closing)
    connected_mask = connect_components(binary_mask, max_dist=25)
    
    # 4. Cleanup (Remove small noise objects)
    try:
        cleaned_mask = remove_small_objects(connected_mask, min_size=100)
    except TypeError:
        cleaned_mask = remove_small_objects(connected_mask, max_size=100)
    
    # 5. Morphological Closing (Smoothing)
    final_mask = closing(cleaned_mask, disk(3))
    
    # 6. Skeletonize
    skeleton = skeletonize(final_mask)
    
    # 7. Prune Skeleton
    final_skeleton = prune_skeleton(skeleton, spur_length=15)
    
    return final_mask, final_skeleton
