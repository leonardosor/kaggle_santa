"""Optimized batch GPU polygon operations with minimal CPU-GPU transfers."""
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = np
    CUDA_AVAILABLE = False


def batch_collision_check_optimized(test_polygons_list, placed_polygons_list, epsilon=1e-6):
    """
    Optimized batch collision detection with minimal CPU-GPU transfers.
    
    Checks all test polygons against all placed polygons in one GPU operation.
    
    Args:
        test_polygons_list: List of N test polygon vertex arrays (each Mx2)
        placed_polygons_list: List of P placed polygon vertex arrays (each Kx2)
        epsilon: Distance threshold for touching detection
        
    Returns:
        NxP array where result[i,j] = True if test[i] collides with placed[j]
    """
    if not CUDA_AVAILABLE or len(test_polygons_list) < 10:
        return batch_collision_check_cpu(test_polygons_list, placed_polygons_list, epsilon)
    
    n_test = len(test_polygons_list)
    n_placed = len(placed_polygons_list)
    
    # Allocate result array on GPU
    collisions = cp.zeros((n_test, n_placed), dtype=cp.bool_)
    
    # Compute bounding boxes for all polygons (on GPU)
    test_bounds = cp.zeros((n_test, 4), dtype=cp.float32)  # minx, miny, maxx, maxy
    placed_bounds = cp.zeros((n_placed, 4), dtype=cp.float32)
    
    for i, poly in enumerate(test_polygons_list):
        poly_gpu = cp.array(poly, dtype=cp.float32)
        test_bounds[i, 0] = cp.min(poly_gpu[:, 0])
        test_bounds[i, 1] = cp.min(poly_gpu[:, 1])
        test_bounds[i, 2] = cp.max(poly_gpu[:, 0])
        test_bounds[i, 3] = cp.max(poly_gpu[:, 1])
    
    for i, poly in enumerate(placed_polygons_list):
        poly_gpu = cp.array(poly, dtype=cp.float32)
        placed_bounds[i, 0] = cp.min(poly_gpu[:, 0])
        placed_bounds[i, 1] = cp.min(poly_gpu[:, 1])
        placed_bounds[i, 2] = cp.max(poly_gpu[:, 0])
        placed_bounds[i, 3] = cp.max(poly_gpu[:, 1])
    
    # Quick rejection using bounding boxes (fully on GPU)
    for i in range(n_test):
        for j in range(n_placed):
            # Check if bounding boxes overlap
            no_overlap = (test_bounds[i, 2] < placed_bounds[j, 0] or  # test right < placed left
                         test_bounds[i, 0] > placed_bounds[j, 2] or  # test left > placed right
                         test_bounds[i, 3] < placed_bounds[j, 1] or  # test top < placed bottom
                         test_bounds[i, 1] > placed_bounds[j, 3])     # test bottom > placed top
            
            if not no_overlap:
                # Need detailed check - mark as potential collision
                collisions[i, j] = True
    
    # Transfer potential collisions to CPU for detailed checking
    potential_collisions = cp.asnumpy(collisions)
    
    # Detailed check only for potential collisions
    detailed_results = np.zeros((n_test, n_placed), dtype=bool)
    
    for i in range(n_test):
        for j in range(n_placed):
            if potential_collisions[i, j]:
                # Do detailed polygon intersection check on CPU
                from gpu_polygon_ops import polygons_intersect_cpu
                intersects, touches = polygons_intersect_cpu(
                    test_polygons_list[i], 
                    placed_polygons_list[j],
                    epsilon
                )
                detailed_results[i, j] = intersects and not touches
    
    return detailed_results


def batch_collision_check_cpu(test_polygons_list, placed_polygons_list, epsilon=1e-6):
    """CPU fallback for batch collision detection."""
    n_test = len(test_polygons_list)
    n_placed = len(placed_polygons_list)
    
    collisions = np.zeros((n_test, n_placed), dtype=bool)
    
    from gpu_polygon_ops import polygons_intersect_cpu
    
    for i in range(n_test):
        for j in range(n_placed):
            intersects, touches = polygons_intersect_cpu(
                test_polygons_list[i],
                placed_polygons_list[j],
                epsilon
            )
            collisions[i, j] = intersects and not touches
    
    return collisions


def optimized_collision_batch_with_caching(test_positions, test_angles, placed_trees, tree_template):
    """
    Highly optimized batch collision detection with GPU caching.
    
    This version:
    1. Caches placed tree polygons on GPU
    2. Processes all test positions in parallel batches
    3. Minimizes CPU-GPU transfers
    
    Args:
        test_positions: Nx2 array of positions to test
        test_angles: N array of angles to test
        placed_trees: List of placed trees
        tree_template: Template tree for creating test polygons
        
    Returns:
        Boolean array of length N indicating collisions
    """
    if not CUDA_AVAILABLE:
        return optimized_collision_batch_cpu(test_positions, test_angles, placed_trees, tree_template)
    
    n_tests = len(test_positions)
    n_placed = len(placed_trees)
    
    if n_placed == 0:
        return np.zeros(n_tests, dtype=bool)
    
    # Cache placed tree data on GPU (done once)
    placed_centers = cp.array(
        [[float(t.center_x), float(t.center_y)] for t in placed_trees],
        dtype=cp.float32
    )
    
    # Move test positions to GPU
    test_pos_gpu = cp.array(test_positions, dtype=cp.float32)
    
    # Compute all pairwise distances on GPU (NxM matrix)
    diff = test_pos_gpu[:, cp.newaxis, :] - placed_centers[cp.newaxis, :, :]
    distances_sq = cp.sum(diff ** 2, axis=2)
    
    # Quick rejection: minimum distance threshold
    # Approximate tree radius as 2.0 for conservative filtering
    threshold_sq = 16.0  # (2 * 2.0)^2
    
    potential_collisions = cp.any(distances_sq < threshold_sq, axis=1)
    potential_indices = cp.where(potential_collisions)[0]
    
    # Transfer to CPU only the indices that need detailed checking
    check_indices = cp.asnumpy(potential_indices)
    
    # Result array
    collisions = np.zeros(n_tests, dtype=bool)
    
    if len(check_indices) == 0:
        return collisions
    
    # Prepare placed polygon vertex arrays (done once)
    from gpu_polygon_ops import polygon_to_array
    placed_poly_verts = [polygon_to_array(t.polygon) for t in placed_trees]
    
    # Process in batches to reduce overhead
    batch_size = min(50, len(check_indices))
    
    for batch_start in range(0, len(check_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(check_indices))
        batch_indices = check_indices[batch_start:batch_end]
        
        # Create test polygons for this batch
        test_poly_verts = []
        for idx in batch_indices:
            from decimal import Decimal
            test_tree = tree_template.copy()
            test_tree.set_position(
                Decimal(str(test_positions[idx, 0])),
                Decimal(str(test_positions[idx, 1])),
                Decimal(str(test_angles[idx]))
            )
            test_poly_verts.append(polygon_to_array(test_tree.polygon))
        
        # Batch collision check
        batch_collisions = batch_collision_check_optimized(test_poly_verts, placed_poly_verts)
        
        # Any collision means this position is invalid
        for i, idx in enumerate(batch_indices):
            if np.any(batch_collisions[i]):
                collisions[idx] = True
    
    return collisions


def optimized_collision_batch_cpu(test_positions, test_angles, placed_trees, tree_template):
    """CPU fallback for optimized batch collision detection."""
    from decimal import Decimal
    from gpu_polygon_ops import polygon_to_array, polygons_intersect_cpu
    
    n_tests = len(test_positions)
    collisions = np.zeros(n_tests, dtype=bool)
    
    # Prepare placed polygons once
    placed_poly_verts = [polygon_to_array(t.polygon) for t in placed_trees]
    
    for i in range(n_tests):
        test_tree = tree_template.copy()
        test_tree.set_position(
            Decimal(str(test_positions[i, 0])),
            Decimal(str(test_positions[i, 1])),
            Decimal(str(test_angles[i]))
        )
        test_poly_verts = polygon_to_array(test_tree.polygon)
        
        for placed_verts in placed_poly_verts:
            intersects, touches = polygons_intersect_cpu(test_poly_verts, placed_verts)
            if intersects and not touches:
                collisions[i] = True
                break
    
    return collisions


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if not CUDA_AVAILABLE:
        return "CUDA not available"
    
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        
        return f"GPU Memory: {used_bytes / 1024**2:.1f} MB used / {total_bytes / 1024**2:.1f} MB total"
    except Exception as e:
        return f"Error getting memory info: {e}"
