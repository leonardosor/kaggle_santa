"""Simple demonstration of GPU polygon operations."""
import numpy as np
from shapely.geometry import Polygon
import gpu_polygon_ops
import time


def demo_basic_operations():
    """Demonstrate basic GPU polygon operations."""
    print("GPU Polygon Operations Demo")
    print("=" * 60)
    print()
    
    # Check if CUDA is available
    try:
        import cupy as cp
        print("✓ CUDA/CuPy is available")
        print(f"  GPU Count: {cp.cuda.runtime.getDeviceCount()}")
    except:
        print("✗ CUDA/CuPy not available - using CPU fallback")
    print()
    
    # Demo 1: Polygon intersection
    print("Demo 1: Polygon Intersection Detection")
    print("-" * 60)
    
    # Two overlapping squares
    square1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    square2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float32)
    
    intersects, touches = gpu_polygon_ops.polygons_intersect_gpu(square1, square2)
    print(f"  Square 1: corners at (0,0) and (10,10)")
    print(f"  Square 2: corners at (5,5) and (15,15)")
    print(f"  Result: intersects={intersects}, touches={touches}")
    print()
    
    # Demo 2: Multiple polygon collision check
    print("Demo 2: Batch Collision Detection")
    print("-" * 60)
    
    # Create a template triangle
    triangle = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.float32)
    
    # Place 10 triangles in a grid
    placed = []
    for i in range(3):
        for j in range(3):
            offset = np.array([i * 3, j * 3], dtype=np.float32)
            placed.append(triangle + offset)
    
    print(f"  Placed {len(placed)} triangles in a 3x3 grid")
    
    # Test 3 candidate positions
    test_positions = [
        triangle + np.array([1, 1], dtype=np.float32),  # Overlaps
        triangle + np.array([10, 10], dtype=np.float32),  # Clear
        triangle + np.array([6, 6], dtype=np.float32),  # Overlaps
    ]
    
    for idx, test_poly in enumerate(test_positions):
        has_collision = False
        for placed_poly in placed:
            intersects, touches = gpu_polygon_ops.polygons_intersect_gpu(test_poly, placed_poly)
            if intersects and not touches:
                has_collision = True
                break
        
        offset = test_poly[0] - triangle[0]
        print(f"  Test position ({offset[0]:.1f}, {offset[1]:.1f}): {'COLLISION' if has_collision else 'CLEAR'}")
    
    print()
    
    # Demo 3: Bounding box computation
    print("Demo 3: Bounding Box Computation")
    print("-" * 60)
    
    polygons = [triangle + np.array([i * 2, i * 2], dtype=np.float32) for i in range(5)]
    minx, miny, maxx, maxy = gpu_polygon_ops.compute_polygon_bounds_gpu(polygons)
    
    print(f"  Computed bounding box for {len(polygons)} polygons")
    print(f"  Bounds: ({minx:.1f}, {miny:.1f}) to ({maxx:.1f}, {maxy:.1f})")
    print(f"  Width: {maxx - minx:.1f}, Height: {maxy - miny:.1f}")
    print()
    
    # Demo 4: Performance comparison
    print("Demo 4: Performance Benchmark")
    print("-" * 60)
    
    # Create test data
    np.random.seed(42)
    n_tests = 50
    n_placed = 20
    
    test_polygons = []
    placed_polygons = []
    
    for i in range(n_tests):
        offset = np.random.randn(2) * 10
        test_polygons.append(triangle + offset)
    
    for i in range(n_placed):
        offset = np.random.randn(2) * 10
        placed_polygons.append(triangle + offset)
    
    # GPU version
    start = time.time()
    gpu_collisions = 0
    for test_poly in test_polygons:
        for placed_poly in placed_polygons:
            intersects, touches = gpu_polygon_ops.polygons_intersect_gpu(test_poly, placed_poly)
            if intersects and not touches:
                gpu_collisions += 1
                break
    gpu_time = time.time() - start
    
    # Shapely version
    test_shapely = [Polygon(p) for p in test_polygons]
    placed_shapely = [Polygon(p) for p in placed_polygons]
    
    start = time.time()
    shapely_collisions = 0
    for test_poly in test_shapely:
        for placed_poly in placed_shapely:
            if test_poly.intersects(placed_poly) and not test_poly.touches(placed_poly):
                shapely_collisions += 1
                break
    shapely_time = time.time() - start
    
    print(f"  Testing {n_tests} positions against {n_placed} placed objects")
    print(f"  GPU time:      {gpu_time*1000:.2f} ms")
    print(f"  Shapely time:  {shapely_time*1000:.2f} ms")
    print(f"  Speedup:       {shapely_time/gpu_time:.2f}x")
    print(f"  Collisions found: {gpu_collisions} (GPU) vs {shapely_collisions} (Shapely)")
    print()
    
    print("=" * 60)
    print("Demo completed!")
    print()
    print("Integration: The GPU polygon operations are automatically")
    print("used in gpu_optimizations.py for collision detection and")
    print("bounding box computation.")


if __name__ == "__main__":
    demo_basic_operations()
