"""Test GPU polygon operations against Shapely to verify correctness."""
import numpy as np
from shapely.geometry import Polygon, Point
import gpu_polygon_ops
import time


def test_point_in_polygon():
    """Test point-in-polygon against Shapely."""
    print("Testing point-in-polygon...")
    
    # Create a simple square polygon
    poly_verts = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float32)
    
    shapely_poly = Polygon(poly_verts)
    
    # Test points
    test_points = np.array([
        [5, 5],    # Inside
        [15, 15],  # Outside
        [0, 0],    # On vertex
        [5, 0],    # On edge
        [-5, 5],   # Outside
    ], dtype=np.float32)
    
    # GPU results
    gpu_inside = gpu_polygon_ops.point_in_polygon_gpu(test_points, poly_verts)
    
    # Shapely results
    shapely_inside = [shapely_poly.contains(Point(p)) for p in test_points]
    
    print(f"  GPU results:     {gpu_inside}")
    print(f"  Shapely results: {shapely_inside}")
    print(f"  Match: {np.allclose(gpu_inside, shapely_inside)}")
    print()


def test_polygon_intersection():
    """Test polygon intersection against Shapely."""
    print("Testing polygon intersection...")
    
    # Polygon 1: Square at origin
    poly1_verts = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float32)
    
    # Polygon 2: Overlapping square
    poly2_verts = np.array([
        [5, 5],
        [15, 5],
        [15, 15],
        [5, 15]
    ], dtype=np.float32)
    
    # Polygon 3: Non-overlapping square
    poly3_verts = np.array([
        [20, 20],
        [30, 20],
        [30, 30],
        [20, 30]
    ], dtype=np.float32)
    
    # Polygon 4: Touching square
    poly4_verts = np.array([
        [10, 0],
        [20, 0],
        [20, 10],
        [10, 10]
    ], dtype=np.float32)
    
    # Create Shapely polygons
    shapely_poly1 = Polygon(poly1_verts)
    shapely_poly2 = Polygon(poly2_verts)
    shapely_poly3 = Polygon(poly3_verts)
    shapely_poly4 = Polygon(poly4_verts)
    
    # Test overlapping
    print("  Test 1: Overlapping polygons")
    gpu_intersects, gpu_touches = gpu_polygon_ops.polygons_intersect_gpu(poly1_verts, poly2_verts)
    shapely_intersects = shapely_poly1.intersects(shapely_poly2) and not shapely_poly1.touches(shapely_poly2)
    print(f"    GPU: intersects={gpu_intersects}, touches={gpu_touches}")
    print(f"    Shapely: intersects={shapely_intersects}")
    print(f"    Match: {gpu_intersects == shapely_intersects}")
    
    # Test non-overlapping
    print("  Test 2: Non-overlapping polygons")
    gpu_intersects, gpu_touches = gpu_polygon_ops.polygons_intersect_gpu(poly1_verts, poly3_verts)
    shapely_intersects = shapely_poly1.intersects(shapely_poly3) and not shapely_poly1.touches(shapely_poly3)
    print(f"    GPU: intersects={gpu_intersects}, touches={gpu_touches}")
    print(f"    Shapely: intersects={shapely_intersects}")
    print(f"    Match: {gpu_intersects == shapely_intersects}")
    
    # Test touching
    print("  Test 3: Touching polygons")
    gpu_intersects, gpu_touches = gpu_polygon_ops.polygons_intersect_gpu(poly1_verts, poly4_verts)
    shapely_touches = shapely_poly1.touches(shapely_poly4)
    print(f"    GPU: intersects={gpu_intersects}, touches={gpu_touches}")
    print(f"    Shapely: touches={shapely_touches}")
    print(f"    Match: {gpu_touches == shapely_touches}")
    print()


def test_bounding_box():
    """Test bounding box computation."""
    print("Testing bounding box computation...")
    
    # Create multiple polygons
    polygons = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
        np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float32),
        np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=np.float32),
    ]
    
    # GPU computation
    gpu_bounds = gpu_polygon_ops.compute_polygon_bounds_gpu(polygons)
    
    # Numpy computation
    all_verts = np.vstack(polygons)
    cpu_bounds = (
        np.min(all_verts[:, 0]),
        np.min(all_verts[:, 1]),
        np.max(all_verts[:, 0]),
        np.max(all_verts[:, 1])
    )
    
    print(f"  GPU bounds:  {gpu_bounds}")
    print(f"  CPU bounds:  {cpu_bounds}")
    print(f"  Match: {np.allclose(gpu_bounds, cpu_bounds)}")
    print()


def benchmark_collision_detection():
    """Benchmark GPU vs Shapely collision detection."""
    print("Benchmarking collision detection...")
    
    # Create a template polygon (triangle)
    template_verts = np.array([
        [0, 0],
        [2, 0],
        [1, 2]
    ], dtype=np.float32)
    
    # Create placed polygons at various positions
    num_placed = 50
    placed_polygons = []
    shapely_placed = []
    
    np.random.seed(42)
    for i in range(num_placed):
        offset = np.random.randn(2) * 10
        verts = template_verts + offset
        placed_polygons.append(verts)
        shapely_placed.append(Polygon(verts))
    
    # Create test polygons
    num_tests = 100
    test_polygons = []
    shapely_tests = []
    
    for i in range(num_tests):
        offset = np.random.randn(2) * 15
        verts = template_verts + offset
        test_polygons.append(verts)
        shapely_tests.append(Polygon(verts))
    
    # Benchmark GPU
    start = time.time()
    gpu_results = []
    for test_verts in test_polygons:
        has_collision = False
        for placed_verts in placed_polygons:
            intersects, touches = gpu_polygon_ops.polygons_intersect_gpu(test_verts, placed_verts)
            if intersects and not touches:
                has_collision = True
                break
        gpu_results.append(has_collision)
    gpu_time = time.time() - start
    
    # Benchmark Shapely
    start = time.time()
    shapely_results = []
    for test_poly in shapely_tests:
        has_collision = False
        for placed_poly in shapely_placed:
            if test_poly.intersects(placed_poly) and not test_poly.touches(placed_poly):
                has_collision = True
                break
        shapely_results.append(has_collision)
    shapely_time = time.time() - start
    
    print(f"  GPU time:     {gpu_time:.4f}s")
    print(f"  Shapely time: {shapely_time:.4f}s")
    print(f"  Speedup:      {shapely_time / gpu_time:.2f}x")
    print(f"  Results match: {gpu_results == shapely_results}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("GPU Polygon Operations Test Suite")
    print("=" * 60)
    print()
    
    try:
        from shapely.geometry import Point as ShapelyPoint
        test_point_in_polygon()
        test_polygon_intersection()
        test_bounding_box()
        benchmark_collision_detection()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
