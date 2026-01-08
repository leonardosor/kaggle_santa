"""GPU-accelerated polygon operations to replace Shapely CPU operations."""
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = np
    CUDA_AVAILABLE = False


# ============================================================================
# GPU Polygon Intersection Detection
# ============================================================================

def point_in_polygon_gpu(points, polygon_vertices):
    """
    Check if points are inside a polygon using ray casting algorithm on GPU.
    
    Args:
        points: Nx2 array of points to test
        polygon_vertices: Mx2 array of polygon vertices
        
    Returns:
        Boolean array of length N indicating if each point is inside
    """
    if not CUDA_AVAILABLE:
        return point_in_polygon_cpu(points, polygon_vertices)
    
    points_gpu = cp.array(points, dtype=cp.float32)
    verts_gpu = cp.array(polygon_vertices, dtype=cp.float32)
    
    n_points = points_gpu.shape[0]
    n_verts = verts_gpu.shape[0]
    
    # Ray casting: count intersections with edges
    # Cast ray from each point to the right (increasing x)
    inside = cp.zeros(n_points, dtype=cp.bool_)
    
    for i in range(n_verts):
        v1 = verts_gpu[i]
        v2 = verts_gpu[(i + 1) % n_verts]
        
        # Check if ray from point crosses edge (v1, v2)
        # Edge must span the point's y coordinate
        crosses_y = ((v1[1] > points_gpu[:, 1]) != (v2[1] > points_gpu[:, 1]))
        
        # Calculate x coordinate of intersection
        x_intersect = (v2[0] - v1[0]) * (points_gpu[:, 1] - v1[1]) / (v2[1] - v1[1] + 1e-10) + v1[0]
        
        # Ray crosses edge if intersection is to the right of point
        crosses = crosses_y & (points_gpu[:, 0] < x_intersect)
        
        # Toggle inside flag for each crossing
        inside = inside ^ crosses
    
    return inside


def point_in_polygon_cpu(points, polygon_vertices):
    """CPU fallback for point-in-polygon test."""
    n_points = len(points)
    n_verts = len(polygon_vertices)
    inside = np.zeros(n_points, dtype=bool)
    
    for i in range(n_verts):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % n_verts]
        
        crosses_y = ((v1[1] > points[:, 1]) != (v2[1] > points[:, 1]))
        x_intersect = (v2[0] - v1[0]) * (points[:, 1] - v1[1]) / (v2[1] - v1[1] + 1e-10) + v1[0]
        crosses = crosses_y & (points[:, 0] < x_intersect)
        inside = inside ^ crosses
    
    return inside


def segments_intersect_gpu(seg1_start, seg1_end, seg2_start, seg2_end):
    """
    Check if two line segments intersect on GPU using cross product method.
    
    Args:
        seg1_start, seg1_end: Start and end points of segment 1 (Nx2 arrays)
        seg2_start, seg2_end: Start and end points of segment 2 (Mx2 arrays)
        
    Returns:
        NxM boolean array indicating intersections
    """
    if not CUDA_AVAILABLE:
        return segments_intersect_cpu(seg1_start, seg1_end, seg2_start, seg2_end)
    
    # Convert to GPU arrays
    s1_start = cp.array(seg1_start, dtype=cp.float32)
    s1_end = cp.array(seg1_end, dtype=cp.float32)
    s2_start = cp.array(seg2_start, dtype=cp.float32)
    s2_end = cp.array(seg2_end, dtype=cp.float32)
    
    # Direction vectors
    d1 = s1_end - s1_start  # (N, 2)
    d2 = s2_end - s2_start  # (M, 2)
    
    # Expand dimensions for broadcasting
    s1_start_exp = s1_start[:, cp.newaxis, :]  # (N, 1, 2)
    d1_exp = d1[:, cp.newaxis, :]              # (N, 1, 2)
    s2_start_exp = s2_start[cp.newaxis, :, :]  # (1, M, 2)
    d2_exp = d2[cp.newaxis, :, :]              # (1, M, 2)
    
    # Cross product d1 x d2
    cross = d1_exp[:, :, 0] * d2_exp[:, :, 1] - d1_exp[:, :, 1] * d2_exp[:, :, 0]
    
    # Parallel segments (cross product = 0)
    parallel = cp.abs(cross) < 1e-10
    
    # For non-parallel segments, compute intersection parameters
    diff = s2_start_exp - s1_start_exp  # (N, M, 2)
    
    # t = (diff x d2) / (d1 x d2)
    # s = (diff x d1) / (d1 x d2)
    t = (diff[:, :, 0] * d2_exp[:, :, 1] - diff[:, :, 1] * d2_exp[:, :, 0]) / (cross + 1e-10)
    s = (diff[:, :, 0] * d1_exp[:, :, 1] - diff[:, :, 1] * d1_exp[:, :, 0]) / (cross + 1e-10)
    
    # Segments intersect if 0 <= t <= 1 and 0 <= s <= 1
    intersects = (~parallel) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    
    return intersects


def segments_intersect_cpu(seg1_start, seg1_end, seg2_start, seg2_end):
    """CPU fallback for segment intersection."""
    n1 = len(seg1_start)
    n2 = len(seg2_start)
    intersects = np.zeros((n1, n2), dtype=bool)
    
    for i in range(n1):
        d1 = seg1_end[i] - seg1_start[i]
        for j in range(n2):
            d2 = seg2_end[j] - seg2_start[j]
            
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) < 1e-10:  # Parallel
                continue
            
            diff = seg2_start[j] - seg1_start[i]
            t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
            s = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
            
            if 0 <= t <= 1 and 0 <= s <= 1:
                intersects[i, j] = True
    
    return intersects


def polygons_intersect_gpu(poly1_verts, poly2_verts, epsilon=1e-6):
    """
    Check if two polygons intersect (overlap, not just touch) on GPU.
    
    Uses:
    1. Fast distance-based filtering
    2. Edge-edge intersection test
    3. Point-in-polygon test (vertices of one inside the other)
    
    Args:
        poly1_verts: Nx2 array of polygon 1 vertices
        poly2_verts: Mx2 array of polygon 2 vertices
        epsilon: Distance threshold for "touching" vs "intersecting"
        
    Returns:
        (intersects, touches) - boolean tuple
    """
    if not CUDA_AVAILABLE:
        return polygons_intersect_cpu(poly1_verts, poly2_verts, epsilon)
    
    poly1 = cp.array(poly1_verts, dtype=cp.float32)
    poly2 = cp.array(poly2_verts, dtype=cp.float32)
    
    n1 = poly1.shape[0]
    n2 = poly2.shape[0]
    
    # Quick rejection: check bounding boxes
    min1 = cp.min(poly1, axis=0)
    max1 = cp.max(poly1, axis=0)
    min2 = cp.min(poly2, axis=0)
    max2 = cp.max(poly2, axis=0)
    
    # If bounding boxes don't overlap, polygons don't intersect
    if float(max1[0]) < float(min2[0]) or float(min1[0]) > float(max2[0]) or \
       float(max1[1]) < float(min2[1]) or float(min1[1]) > float(max2[1]):
        return False, False
    
    # Test 1: Check for proper edge intersections (not at endpoints)
    edge_intersects = False
    closest_dist_sq = float('inf')
    
    # Convert to CPU for detailed checks (more efficient for small polygon counts)
    poly1_cpu = cp.asnumpy(poly1)
    poly2_cpu = cp.asnumpy(poly2)
    
    for i in range(n1):
        seg1_start = poly1_cpu[i]
        seg1_end = poly1_cpu[(i + 1) % n1]
        
        for j in range(n2):
            seg2_start = poly2_cpu[j]
            seg2_end = poly2_cpu[(j + 1) % n2]
            
            # Compute minimum distance between edges
            # Distance between segment midpoints as approximation
            mid1 = (seg1_start + seg1_end) / 2
            mid2 = (seg2_start + seg2_end) / 2
            dist_sq = np.sum((mid1 - mid2) ** 2)
            closest_dist_sq = min(closest_dist_sq, dist_sq)
            
            # Check for proper intersection
            d1 = seg1_end - seg1_start
            d2 = seg2_end - seg2_start
            
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) > 1e-10:  # Not parallel
                diff = seg2_start - seg1_start
                t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
                s = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
                
                # Proper intersection (not at endpoints)
                if 0.01 < t < 0.99 and 0.01 < s < 0.99:
                    edge_intersects = True
                    break
        
        if edge_intersects:
            break
    
    # Test 2: Check if any vertices of poly1 are inside poly2
    inside1 = point_in_polygon_cpu(poly1_cpu, poly2_cpu)
    any_inside1 = np.any(inside1)
    
    # Test 3: Check if any vertices of poly2 are inside poly1
    inside2 = point_in_polygon_cpu(poly2_cpu, poly1_cpu)
    any_inside2 = np.any(inside2)
    
    # Polygons intersect if:
    # - Edges cross properly, OR
    # - One polygon contains vertices of the other
    intersects = edge_intersects or any_inside1 or any_inside2
    
    # Polygons touch (but don't overlap) if very close but don't intersect
    touches = (closest_dist_sq < epsilon ** 2) and not intersects
    
    return intersects, touches


def polygons_intersect_cpu(poly1_verts, poly2_verts, epsilon=1e-6):
    """CPU fallback for polygon intersection."""
    n1 = len(poly1_verts)
    n2 = len(poly2_verts)
    
    # Check edge intersections
    edge_intersects = False
    min_edge_dist = float('inf')
    
    for i in range(n1):
        seg1_start = poly1_verts[i]
        seg1_end = poly1_verts[(i + 1) % n1]
        
        for j in range(n2):
            seg2_start = poly2_verts[j]
            seg2_end = poly2_verts[(j + 1) % n2]
            
            # Distance check
            mid1 = (seg1_start + seg1_end) / 2
            mid2 = (seg2_start + seg2_end) / 2
            dist = np.linalg.norm(mid1 - mid2)
            min_edge_dist = min(min_edge_dist, dist)
            
            # Intersection check
            d1 = seg1_end - seg1_start
            d2 = seg2_end - seg2_start
            
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) > 1e-10:
                diff = seg2_start - seg1_start
                t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
                s = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
                
                if 0 < t < 1 and 0 < s < 1:
                    edge_intersects = True
                    break
        
        if edge_intersects:
            break
    
    # Point-in-polygon tests
    inside1 = point_in_polygon_cpu(poly1_verts, poly2_verts)
    inside2 = point_in_polygon_cpu(poly2_verts, poly1_verts)
    
    intersects = edge_intersects or np.any(inside1) or np.any(inside2)
    touches = (min_edge_dist < epsilon) and not intersects
    
    return intersects, touches


def batch_polygon_intersections_gpu(test_polygons, placed_polygons, epsilon=1e-6):
    """
    Check intersections between multiple test polygons and placed polygons.
    
    Args:
        test_polygons: List of N test polygon vertex arrays
        placed_polygons: List of M placed polygon vertex arrays
        epsilon: Touch threshold
        
    Returns:
        NxM array of (intersects, touches) tuples
    """
    if not CUDA_AVAILABLE:
        return batch_polygon_intersections_cpu(test_polygons, placed_polygons, epsilon)
    
    n_test = len(test_polygons)
    n_placed = len(placed_polygons)
    
    results = np.zeros((n_test, n_placed), dtype=[('intersects', bool), ('touches', bool)])
    
    # Process each test polygon
    for i in range(n_test):
        test_verts = test_polygons[i]
        
        # Check against all placed polygons
        for j in range(n_placed):
            placed_verts = placed_polygons[j]
            
            intersects, touches = polygons_intersect_gpu(test_verts, placed_verts, epsilon)
            results[i, j]['intersects'] = intersects
            results[i, j]['touches'] = touches
    
    return results


def batch_polygon_intersections_cpu(test_polygons, placed_polygons, epsilon=1e-6):
    """CPU fallback for batch polygon intersections."""
    n_test = len(test_polygons)
    n_placed = len(placed_polygons)
    
    results = np.zeros((n_test, n_placed), dtype=[('intersects', bool), ('touches', bool)])
    
    for i in range(n_test):
        for j in range(n_placed):
            intersects, touches = polygons_intersect_cpu(
                test_polygons[i], placed_polygons[j], epsilon
            )
            results[i, j]['intersects'] = intersects
            results[i, j]['touches'] = touches
    
    return results


# ============================================================================
# GPU Bounding Box Computation
# ============================================================================

def compute_polygon_bounds_gpu(polygons):
    """
    Compute bounding box for a list of polygons on GPU.
    
    Args:
        polygons: List of polygon vertex arrays (each Nx2)
        
    Returns:
        (minx, miny, maxx, maxy) tuple
    """
    if not CUDA_AVAILABLE or len(polygons) == 0:
        return compute_polygon_bounds_cpu(polygons)
    
    # Concatenate all vertices
    all_verts = np.vstack(polygons)
    verts_gpu = cp.array(all_verts, dtype=cp.float32)
    
    # Compute bounds
    minx = float(cp.min(verts_gpu[:, 0]))
    miny = float(cp.min(verts_gpu[:, 1]))
    maxx = float(cp.max(verts_gpu[:, 0]))
    maxy = float(cp.max(verts_gpu[:, 1]))
    
    return minx, miny, maxx, maxy


def compute_polygon_bounds_cpu(polygons):
    """CPU fallback for bounding box computation."""
    if len(polygons) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    all_verts = np.vstack(polygons)
    
    minx = np.min(all_verts[:, 0])
    miny = np.min(all_verts[:, 1])
    maxx = np.max(all_verts[:, 0])
    maxy = np.max(all_verts[:, 1])
    
    return minx, miny, maxx, maxy


# ============================================================================
# Helper Functions
# ============================================================================

def polygon_to_array(shapely_polygon):
    """Convert Shapely polygon to numpy array of vertices."""
    coords = list(shapely_polygon.exterior.coords[:-1])  # Exclude duplicate last point
    return np.array(coords, dtype=np.float32)


def arrays_to_shapely_polygon(vertices):
    """Convert numpy array to Shapely polygon (for compatibility)."""
    from shapely.geometry import Polygon
    return Polygon(vertices)
