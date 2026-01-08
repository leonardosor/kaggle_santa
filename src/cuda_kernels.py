"""Custom CUDA kernels for high-performance collision detection."""
import numpy as np
import sys

print("\n" + "="*60)
print("[CUDA] Kernels Module Loading...")
print("="*60)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("[OK] CuPy imported successfully")
except ImportError:
    cp = np
    CUDA_AVAILABLE = False
    print("[WARN] CuPy not available - CUDA kernels disabled")
    print("="*60 + "\n")


# CUDA kernel for point-in-polygon test using ray casting
POINT_IN_POLYGON_KERNEL = r"""
extern "C" __global__
void point_in_polygon(
    const float* points,      // Nx2 array of test points
    const float* polygon,     // Mx2 array of polygon vertices
    bool* results,            // N array of results
    int n_points,
    int n_vertices
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    float px = points[idx * 2];
    float py = points[idx * 2 + 1];
    
    bool inside = false;
    
    // Ray casting algorithm
    for (int i = 0, j = n_vertices - 1; i < n_vertices; j = i++) {
        float vix = polygon[i * 2];
        float viy = polygon[i * 2 + 1];
        float vjx = polygon[j * 2];
        float vjy = polygon[j * 2 + 1];
        
        // Check if ray crosses edge
        if (((viy > py) != (vjy > py)) &&
            (px < (vjx - vix) * (py - viy) / (vjy - viy + 1e-10f) + vix)) {
            inside = !inside;
        }
    }
    
    results[idx] = inside;
}
"""


# CUDA kernel for bounding box overlap check
BBOX_OVERLAP_KERNEL = r"""
extern "C" __global__
void bbox_overlap(
    const float* bounds1,     // Nx4 array (minx, miny, maxx, maxy)
    const float* bounds2,     // Mx4 array (minx, miny, maxx, maxy)
    bool* results,            // NxM array of results
    int n1,
    int n2
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i >= n1 || j >= n2) return;
    
    float min1x = bounds1[i * 4];
    float min1y = bounds1[i * 4 + 1];
    float max1x = bounds1[i * 4 + 2];
    float max1y = bounds1[i * 4 + 3];
    
    float min2x = bounds2[j * 4];
    float min2y = bounds2[j * 4 + 1];
    float max2x = bounds2[j * 4 + 2];
    float max2y = bounds2[j * 4 + 3];
    
    // Boxes overlap if they don't NOT overlap
    bool overlap = !(max1x < min2x || min1x > max2x || 
                     max1y < min2y || min1y > max2y);
    
    results[i * n2 + j] = overlap;
}
"""


# CUDA kernel for line segment intersection
SEGMENT_INTERSECTION_KERNEL = r"""
extern "C" __global__
void segment_intersection(
    const float* seg1_start,  // Nx2 array
    const float* seg1_end,    // Nx2 array
    const float* seg2_start,  // Mx2 array
    const float* seg2_end,    // Mx2 array
    bool* results,            // NxM array
    int n1,
    int n2
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i >= n1 || j >= n2) return;
    
    float p1x = seg1_start[i * 2];
    float p1y = seg1_start[i * 2 + 1];
    float p2x = seg1_end[i * 2];
    float p2y = seg1_end[i * 2 + 1];
    
    float p3x = seg2_start[j * 2];
    float p3y = seg2_start[j * 2 + 1];
    float p4x = seg2_end[j * 2];
    float p4y = seg2_end[j * 2 + 1];
    
    // Direction vectors
    float d1x = p2x - p1x;
    float d1y = p2y - p1y;
    float d2x = p4x - p3x;
    float d2y = p4y - p3y;
    
    // Cross product
    float cross = d1x * d2y - d1y * d2x;
    
    bool intersects = false;
    
    if (fabsf(cross) > 1e-10f) {  // Not parallel
        float diffx = p3x - p1x;
        float diffy = p3y - p1y;
        
        float t = (diffx * d2y - diffy * d2x) / cross;
        float s = (diffx * d1y - diffy * d1x) / cross;
        
        // Proper intersection (not at endpoints)
        if (t > 0.01f && t < 0.99f && s > 0.01f && s < 0.99f) {
            intersects = true;
        }
    }
    
    results[i * n2 + j] = intersects;
}
"""


# CUDA kernel for distance-based filtering
DISTANCE_FILTER_KERNEL = r"""
extern "C" __global__
void distance_filter(
    const float* points1,     // Nx2 array
    const float* points2,     // Mx2 array
    bool* results,            // NxM array
    float threshold_sq,
    int n1,
    int n2
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i >= n1 || j >= n2) return;
    
    float dx = points1[i * 2] - points2[j * 2];
    float dy = points1[i * 2 + 1] - points2[j * 2 + 1];
    float dist_sq = dx * dx + dy * dy;
    
    results[i * n2 + j] = (dist_sq < threshold_sq);
}
"""


class CUDAKernels:
    """Compiled CUDA kernels for polygon operations."""
    
    def __init__(self):
        """Initialize and compile CUDA kernels."""
        self.available = CUDA_AVAILABLE
        
        if not self.available:
            print("[WARN] CUDA not available - skipping kernel compilation")
            print("="*60 + "\n")
            sys.stdout.flush()
            return
        
        print("\n[CUDA] Compiling CUDA kernels (one-time setup)...")
        sys.stdout.flush()
        
        try:
            # Compile kernels
            self.point_in_polygon_kernel = cp.RawKernel(
                POINT_IN_POLYGON_KERNEL, 
                'point_in_polygon'
            )
            print("   [OK] Point-in-polygon kernel compiled")
            
            self.bbox_overlap_kernel = cp.RawKernel(
                BBOX_OVERLAP_KERNEL,
                'bbox_overlap'
            )
            print("   [OK] Bounding box overlap kernel compiled")
            
            self.segment_intersection_kernel = cp.RawKernel(
                SEGMENT_INTERSECTION_KERNEL,
                'segment_intersection'
            )
            print("   [OK] Segment intersection kernel compiled")
            
            self.distance_filter_kernel = cp.RawKernel(
                DISTANCE_FILTER_KERNEL,
                'distance_filter'
            )
            print("   [OK] Distance filter kernel compiled")
            
            print("\n[SUCCESS] CUDA kernels ready! Using maximum performance mode.")
            print("="*60 + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n[ERROR] CUDA kernel compilation failed: {e}")
            print("   Falling back to standard GPU operations.")
            print("="*60 + "\n")
            sys.stdout.flush()
            self.available = False
            raise
    
    def point_in_polygon(self, points, polygon):
        """
        Check if points are inside polygon using CUDA kernel.
        
        Args:
            points: Nx2 CuPy array of test points
            polygon: Mx2 CuPy array of polygon vertices
            
        Returns:
            N boolean CuPy array
        """
        if not self.available:
            raise RuntimeError("CUDA not available")
        
        n_points = points.shape[0]
        n_vertices = polygon.shape[0]
        
        # Allocate result array
        results = cp.zeros(n_points, dtype=cp.bool_)
        
        # Launch kernel
        block_size = 256
        grid_size = (n_points + block_size - 1) // block_size
        
        self.point_in_polygon_kernel(
            (grid_size,), (block_size,),
            (points, polygon, results, n_points, n_vertices)
        )
        
        return results
    
    def bbox_overlap(self, bounds1, bounds2):
        """
        Check bounding box overlaps using CUDA kernel.
        
        Args:
            bounds1: Nx4 CuPy array (minx, miny, maxx, maxy)
            bounds2: Mx4 CuPy array (minx, miny, maxx, maxy)
            
        Returns:
            NxM boolean CuPy array
        """
        if not self.available:
            raise RuntimeError("CUDA not available")
        
        n1 = bounds1.shape[0]
        n2 = bounds2.shape[0]
        
        results = cp.zeros((n1, n2), dtype=cp.bool_)
        
        # 2D grid for pairwise comparisons
        block_size = (16, 16)
        grid_size = (
            (n1 + block_size[0] - 1) // block_size[0],
            (n2 + block_size[1] - 1) // block_size[1]
        )
        
        self.bbox_overlap_kernel(
            grid_size, block_size,
            (bounds1, bounds2, results, n1, n2)
        )
        
        return results
    
    def distance_filter(self, points1, points2, threshold):
        """
        Filter point pairs by distance using CUDA kernel.
        
        Args:
            points1: Nx2 CuPy array
            points2: Mx2 CuPy array
            threshold: Distance threshold
            
        Returns:
            NxM boolean CuPy array (True if distance < threshold)
        """
        if not self.available:
            raise RuntimeError("CUDA not available")
        
        n1 = points1.shape[0]
        n2 = points2.shape[0]
        
        results = cp.zeros((n1, n2), dtype=cp.bool_)
        threshold_sq = float(threshold ** 2)
        
        block_size = (16, 16)
        grid_size = (
            (n1 + block_size[0] - 1) // block_size[0],
            (n2 + block_size[1] - 1) // block_size[1]
        )
        
        self.distance_filter_kernel(
            grid_size, block_size,
            (points1, points2, results, threshold_sq, n1, n2)
        )
        
        return results


# Global kernel instance
_cuda_kernels = None


def get_cuda_kernels():
    """Get or create global CUDA kernels instance."""
    global _cuda_kernels
    if _cuda_kernels is None and CUDA_AVAILABLE:
        _cuda_kernels = CUDAKernels()
    return _cuda_kernels


def cuda_batch_collision_check(test_positions, test_angles, placed_trees, tree_template):
    """
    Ultra-fast batch collision detection using custom CUDA kernels.
    
    Args:
        test_positions: Nx2 array of test positions
        test_angles: N array of test angles
        placed_trees: List of placed trees
        tree_template: Template tree
        
    Returns:
        Boolean array indicating collisions
    """
    if not CUDA_AVAILABLE:
        from gpu_polygon_batch import optimized_collision_batch_cpu
        return optimized_collision_batch_cpu(test_positions, test_angles, placed_trees, tree_template)
    
    kernels = get_cuda_kernels()
    if kernels is None:
        from gpu_polygon_batch import optimized_collision_batch_cpu
        return optimized_collision_batch_cpu(test_positions, test_angles, placed_trees, tree_template)
    
    n_tests = len(test_positions)
    n_placed = len(placed_trees)
    
    if n_placed == 0:
        return np.zeros(n_tests, dtype=bool)
    
    # Transfer test positions to GPU
    test_pos_gpu = cp.array(test_positions, dtype=cp.float32)
    
    # Get placed tree centers
    placed_centers = cp.array(
        [[float(t.center_x), float(t.center_y)] for t in placed_trees],
        dtype=cp.float32
    )
    
    # Use CUDA kernel for distance filtering
    close_pairs = kernels.distance_filter(test_pos_gpu, placed_centers, 4.0)
    
    # Check which test positions need detailed checking
    needs_check = cp.any(close_pairs, axis=1)
    check_indices = cp.asnumpy(cp.where(needs_check)[0])
    
    collisions = np.zeros(n_tests, dtype=bool)
    
    if len(check_indices) == 0:
        return collisions
    
    # For potential collisions, do detailed check on CPU
    # (Full GPU polygon intersection would require more complex kernels)
    from gpu_polygon_ops import polygon_to_array, polygons_intersect_cpu
    from decimal import Decimal
    
    placed_poly_verts = [polygon_to_array(t.polygon) for t in placed_trees]
    
    for idx in check_indices:
        test_tree = tree_template.copy()
        test_tree.set_position(
            Decimal(str(test_positions[idx, 0])),
            Decimal(str(test_positions[idx, 1])),
            Decimal(str(test_angles[idx]))
        )
        
        test_poly_verts = polygon_to_array(test_tree.polygon)
        
        for placed_verts in placed_poly_verts:
            intersects, touches = polygons_intersect_cpu(test_poly_verts, placed_verts)
            if intersects and not touches:
                collisions[idx] = True
                break
    
    return collisions


def benchmark_cuda_kernels():
    """Benchmark CUDA kernels vs Python implementation."""
    if not CUDA_AVAILABLE:
        print("CUDA not available")
        return
    
    print("Benchmarking CUDA Kernels")
    print("=" * 60)
    
    kernels = get_cuda_kernels()
    
    # Test 1: Point-in-polygon
    print("\nTest 1: Point-in-Polygon")
    print("-" * 60)
    
    polygon = cp.array([
        [0, 0], [10, 0], [10, 10], [0, 10]
    ], dtype=cp.float32)
    
    points = cp.random.randn(10000, 2, dtype=cp.float32) * 15
    
    import time
    
    # Warm up
    _ = kernels.point_in_polygon(points[:100], polygon)
    
    start = time.time()
    result = kernels.point_in_polygon(points, polygon)
    kernel_time = time.time() - start
    
    print(f"  CUDA kernel: {kernel_time*1000:.2f} ms for {len(points)} points")
    print(f"  Inside: {cp.sum(result).get()} points")
    
    # Test 2: Distance filtering
    print("\nTest 2: Distance Filtering")
    print("-" * 60)
    
    points1 = cp.random.randn(1000, 2, dtype=cp.float32) * 10
    points2 = cp.random.randn(50, 2, dtype=cp.float32) * 10
    
    start = time.time()
    close_pairs = kernels.distance_filter(points1, points2, 2.0)
    kernel_time = time.time() - start
    
    print(f"  CUDA kernel: {kernel_time*1000:.2f} ms for {len(points1)}x{len(points2)} pairs")
    print(f"  Close pairs: {cp.sum(close_pairs).get()}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_cuda_kernels()
