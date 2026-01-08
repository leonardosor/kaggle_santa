"""GPU memory pool and caching for efficient polygon operations."""
import numpy as np
from collections import OrderedDict
import hashlib

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = np
    CUDA_AVAILABLE = False


class GPUMemoryPool:
    """Manages GPU memory pool and caches polygon data."""
    
    def __init__(self, max_cache_size=1000):
        """
        Initialize GPU memory pool.
        
        Args:
            max_cache_size: Maximum number of polygon vertex arrays to cache
        """
        self.max_cache_size = max_cache_size
        self.polygon_cache = OrderedDict()  # LRU cache for polygon vertices on GPU
        self.mempool = None
        self.pinned_mempool = None
        
        if CUDA_AVAILABLE:
            # Use CuPy's memory pool for faster allocations
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Pre-allocate some memory to avoid initial overhead
            self._preallocate_memory()
    
    def _preallocate_memory(self):
        """Pre-allocate GPU memory to reduce allocation overhead."""
        if not CUDA_AVAILABLE:
            return
        
        # Allocate and free some arrays to warm up the memory pool
        sizes = [100, 1000, 10000]
        for size in sizes:
            temp = cp.zeros((size, 2), dtype=cp.float32)
            del temp
    
    def _compute_hash(self, polygon_array):
        """Compute hash for polygon array to use as cache key."""
        # Use first few and last few points for quick hash
        if len(polygon_array) > 10:
            key_points = np.concatenate([polygon_array[:5], polygon_array[-5:]])
        else:
            key_points = polygon_array
        
        # Convert to bytes and hash
        return hashlib.md5(key_points.tobytes()).hexdigest()
    
    def get_gpu_array(self, polygon_array, cache_key=None):
        """
        Get polygon vertex array on GPU, using cache if available.
        
        Args:
            polygon_array: Numpy array of polygon vertices
            cache_key: Optional cache key (uses hash if not provided)
            
        Returns:
            CuPy array on GPU
        """
        if not CUDA_AVAILABLE:
            return polygon_array
        
        # Generate cache key
        if cache_key is None:
            cache_key = self._compute_hash(polygon_array)
        
        # Check cache
        if cache_key in self.polygon_cache:
            # Move to end (most recently used)
            self.polygon_cache.move_to_end(cache_key)
            return self.polygon_cache[cache_key]
        
        # Not in cache, transfer to GPU
        gpu_array = cp.array(polygon_array, dtype=cp.float32)
        
        # Add to cache
        self.polygon_cache[cache_key] = gpu_array
        
        # Enforce cache size limit (LRU eviction)
        if len(self.polygon_cache) > self.max_cache_size:
            # Remove oldest item
            self.polygon_cache.popitem(last=False)
        
        return gpu_array
    
    def get_batch_gpu_arrays(self, polygon_list):
        """
        Transfer a batch of polygons to GPU with caching.
        
        Args:
            polygon_list: List of polygon vertex arrays
            
        Returns:
            List of CuPy arrays on GPU
        """
        if not CUDA_AVAILABLE:
            return polygon_list
        
        gpu_arrays = []
        for poly in polygon_list:
            gpu_array = self.get_gpu_array(poly)
            gpu_arrays.append(gpu_array)
        
        return gpu_arrays
    
    def clear_cache(self):
        """Clear the polygon cache."""
        self.polygon_cache.clear()
    
    def get_memory_info(self):
        """Get GPU memory usage information."""
        if not CUDA_AVAILABLE:
            return {
                'used_bytes': 0,
                'total_bytes': 0,
                'cache_size': 0
            }
        
        used_bytes = self.mempool.used_bytes()
        total_bytes = self.mempool.total_bytes()
        
        return {
            'used_bytes': used_bytes,
            'used_mb': used_bytes / (1024**2),
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024**2),
            'cache_size': len(self.polygon_cache),
            'cache_limit': self.max_cache_size
        }
    
    def free_unused_memory(self):
        """Free unused GPU memory blocks."""
        if CUDA_AVAILABLE and self.mempool:
            self.mempool.free_all_blocks()
            if self.pinned_mempool:
                self.pinned_mempool.free_all_blocks()


# Global memory pool instance
_global_memory_pool = None


def get_memory_pool():
    """Get or create the global GPU memory pool."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = GPUMemoryPool(max_cache_size=1000)
    return _global_memory_pool


def reset_memory_pool():
    """Reset the global memory pool (useful for testing)."""
    global _global_memory_pool
    if _global_memory_pool:
        _global_memory_pool.clear_cache()
        _global_memory_pool.free_unused_memory()
    _global_memory_pool = None


class CachedPolygonBatch:
    """Manages a cached batch of polygons on GPU."""
    
    def __init__(self, polygon_list):
        """
        Create a cached batch of polygons.
        
        Args:
            polygon_list: List of polygon vertex arrays
        """
        self.memory_pool = get_memory_pool()
        self.polygon_list = polygon_list
        self.gpu_arrays = None
        self.bounds_gpu = None
        self.centers_gpu = None
        
        # Pre-compute and cache on GPU
        self._prepare_gpu_data()
    
    def _prepare_gpu_data(self):
        """Prepare and cache polygon data on GPU."""
        if not CUDA_AVAILABLE:
            return
        
        # Transfer polygons to GPU with caching
        self.gpu_arrays = self.memory_pool.get_batch_gpu_arrays(self.polygon_list)
        
        # Pre-compute bounding boxes on GPU
        n_polys = len(self.gpu_arrays)
        self.bounds_gpu = cp.zeros((n_polys, 4), dtype=cp.float32)  # minx, miny, maxx, maxy
        self.centers_gpu = cp.zeros((n_polys, 2), dtype=cp.float32)
        
        for i, poly_gpu in enumerate(self.gpu_arrays):
            self.bounds_gpu[i, 0] = cp.min(poly_gpu[:, 0])
            self.bounds_gpu[i, 1] = cp.min(poly_gpu[:, 1])
            self.bounds_gpu[i, 2] = cp.max(poly_gpu[:, 0])
            self.bounds_gpu[i, 3] = cp.max(poly_gpu[:, 1])
            
            self.centers_gpu[i, 0] = cp.mean(poly_gpu[:, 0])
            self.centers_gpu[i, 1] = cp.mean(poly_gpu[:, 1])
    
    def get_bounds(self):
        """Get bounding boxes (on GPU)."""
        return self.bounds_gpu
    
    def get_centers(self):
        """Get polygon centers (on GPU)."""
        return self.centers_gpu
    
    def get_polygons(self):
        """Get polygon vertex arrays (on GPU)."""
        return self.gpu_arrays
    
    def __len__(self):
        """Number of polygons in batch."""
        return len(self.polygon_list)


def optimized_batch_collision_with_memory_pool(test_positions, test_angles, placed_trees, tree_template):
    """
    Optimized batch collision detection using memory pool and caching.
    
    Args:
        test_positions: Nx2 array of positions to test
        test_angles: N array of angles to test
        placed_trees: List of placed trees
        tree_template: Template tree
        
    Returns:
        Boolean array indicating collisions
    """
    if not CUDA_AVAILABLE or len(placed_trees) == 0:
        from gpu_polygon_batch import optimized_collision_batch_cpu
        return optimized_collision_batch_cpu(test_positions, test_angles, placed_trees, tree_template)
    
    n_tests = len(test_positions)
    n_placed = len(placed_trees)
    
    memory_pool = get_memory_pool()
    
    # Create cached batch of placed polygons (reused across multiple checks)
    from gpu_polygon_ops import polygon_to_array
    placed_poly_verts = [polygon_to_array(t.polygon) for t in placed_trees]
    placed_batch = CachedPolygonBatch(placed_poly_verts)
    
    # Move test positions to GPU once
    test_pos_gpu = cp.array(test_positions, dtype=cp.float32)
    
    # Use cached centers for distance computation
    placed_centers = placed_batch.get_centers()
    
    # Compute all pairwise distances on GPU
    diff = test_pos_gpu[:, cp.newaxis, :] - placed_centers[cp.newaxis, :, :]
    distances_sq = cp.sum(diff ** 2, axis=2)
    
    # Quick rejection using cached bounding boxes
    threshold_sq = 16.0
    potential_collisions = cp.any(distances_sq < threshold_sq, axis=1)
    check_indices = cp.asnumpy(cp.where(potential_collisions)[0])
    
    # Result array
    collisions = np.zeros(n_tests, dtype=bool)
    
    if len(check_indices) == 0:
        return collisions
    
    # Detailed collision check for potential candidates
    from gpu_polygon_ops import polygons_intersect_cpu
    from decimal import Decimal
    
    # Process in batches
    batch_size = 50
    for batch_start in range(0, len(check_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(check_indices))
        batch_indices = check_indices[batch_start:batch_end]
        
        # Create test polygons for this batch
        for idx in batch_indices:
            test_tree = tree_template.copy()
            test_tree.set_position(
                Decimal(str(test_positions[idx, 0])),
                Decimal(str(test_positions[idx, 1])),
                Decimal(str(test_angles[idx]))
            )
            
            # Use cached GPU arrays for placed polygons
            test_poly_verts = polygon_to_array(test_tree.polygon)
            
            # Check against all placed polygons
            for placed_verts in placed_poly_verts:
                intersects, touches = polygons_intersect_cpu(test_poly_verts, placed_verts)
                if intersects and not touches:
                    collisions[idx] = True
                    break
    
    return collisions


def print_memory_stats():
    """Print GPU memory pool statistics."""
    memory_pool = get_memory_pool()
    stats = memory_pool.get_memory_info()
    
    print("\nGPU Memory Pool Statistics:")
    print("=" * 60)
    print(f"Used memory:     {stats['used_mb']:.2f} MB")
    print(f"Total allocated: {stats['total_mb']:.2f} MB")
    print(f"Cache size:      {stats['cache_size']} / {stats['cache_limit']} polygons")
    print("=" * 60)
