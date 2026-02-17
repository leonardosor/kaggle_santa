"""Test Go geometry integration with Python."""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import go_geometry

def test_point_in_polygon():
    """Test point-in-polygon functionality."""
    print("Testing point-in-polygon...")
    
    # Square polygon
    square = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float64)
    
    # Test points
    tests = [
        (np.array([5, 5]), True, "Inside"),
        (np.array([15, 15]), False, "Outside"),
        (np.array([-5, 5]), False, "Outside left"),
    ]
    
    passed = 0
    for point, expected, desc in tests:
        result = go_geometry.point_in_polygon_go(point, square)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: point={point}, expected={expected}, got={result}")
        if result == expected:
            passed += 1
    
    print(f"  Passed {passed}/{len(tests)} tests\n")
    return passed == len(tests)


def test_segments_intersect():
    """Test segment intersection."""
    print("Testing segment intersection...")
    
    tests = [
        # Crossing segments
        (np.array([0, 0]), np.array([10, 10]),
         np.array([0, 10]), np.array([10, 0]),
         True, "Crossing"),
        # Parallel segments
        (np.array([0, 0]), np.array([10, 0]),
         np.array([0, 5]), np.array([10, 5]),
         False, "Parallel"),
    ]
    
    passed = 0
    for s1_start, s1_end, s2_start, s2_end, expected, desc in tests:
        result = go_geometry.segments_intersect_go(s1_start, s1_end, s2_start, s2_end)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: expected={expected}, got={result}")
        if result == expected:
            passed += 1
    
    print(f"  Passed {passed}/{len(tests)} tests\n")
    return passed == len(tests)


def test_polygons_intersect():
    """Test polygon intersection."""
    print("Testing polygon intersection...")
    
    # Square at origin
    square1 = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float64)
    
    tests = [
        # Overlapping square
        (np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64),
         True, "Overlapping"),
        # Non-overlapping square
        (np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64),
         False, "Non-overlapping"),
        # Contained square
        (np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float64),
         True, "Contained"),
    ]
    
    passed = 0
    for square2, expected, desc in tests:
        intersects, touches = go_geometry.polygons_intersect_go(square1, square2)
        result = intersects
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: expected={expected}, got={result}")
        if result == expected:
            passed += 1
    
    print(f"  Passed {passed}/{len(tests)} tests\n")
    return passed == len(tests)


def test_rotate_and_translate():
    """Test polygon transformation."""
    print("Testing rotate and translate...")
    
    square = np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ], dtype=np.float64)
    
    # Rotate 90 degrees and translate
    result = go_geometry.rotate_and_translate_polygon_go(
        square, np.pi/2, 10, 10
    )
    
    # First vertex should be approximately (10, 11) after rotation and translation
    expected_first = np.array([10, 11])
    diff = np.abs(result[0] - expected_first)
    
    passed = np.all(diff < 1e-6)
    status = "✓" if passed else "✗"
    print(f"  {status} First vertex: expected≈{expected_first}, got={result[0]}")
    print(f"  Passed {1 if passed else 0}/1 tests\n")
    return passed


def test_bounding_box():
    """Test bounding box computation."""
    print("Testing bounding box...")
    
    polygon = np.array([
        [0, 0],
        [10, 5],
        [5, 10],
        [-2, 3]
    ], dtype=np.float64)
    
    minX, minY, maxX, maxY = go_geometry.get_bounding_box_go(polygon)
    
    expected = (-2, 0, 10, 10)
    result = (minX, minY, maxX, maxY)
    
    passed = result == expected
    status = "✓" if passed else "✗"
    print(f"  {status} BBox: expected={expected}, got={result}")
    print(f"  Passed {1 if passed else 0}/1 tests\n")
    return passed


def test_batch_collision():
    """Test batch collision detection."""
    print("Testing batch collision detection...")
    
    # Placed polygon
    placed = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    ]
    
    # Test polygons: one overlapping, one not
    test_polys = [
        np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64),  # Overlapping
        np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64),  # Not overlapping
    ]
    
    results = go_geometry.batch_collision_check_go(test_polys, placed)
    
    expected = np.array([True, False])
    passed = np.all(results == expected)
    
    status = "✓" if passed else "✗"
    print(f"  {status} Collision results: expected={expected}, got={results}")
    print(f"  Passed {1 if passed else 0}/1 tests\n")
    return passed


def benchmark_comparison():
    """Benchmark Go vs pure Python performance."""
    print("Running performance benchmark...")
    print("=" * 60)
    
    # Create test data
    num_tests = 100
    num_placed = 10
    
    np.random.seed(42)
    test_polys = []
    for i in range(num_tests):
        x = np.random.rand() * 100
        y = np.random.rand() * 100
        test_polys.append(np.array([
            [x, y],
            [x+5, y],
            [x+5, y+5],
            [x, y+5]
        ], dtype=np.float64))
    
    placed_polys = []
    for i in range(num_placed):
        x = i * 15.0
        placed_polys.append(np.array([
            [x, 0],
            [x+10, 0],
            [x+10, 10],
            [x, 10]
        ], dtype=np.float64))
    
    # Benchmark Go implementation
    start = time.time()
    for _ in range(10):
        results_go = go_geometry.batch_collision_check_go(test_polys, placed_polys)
    go_time = (time.time() - start) / 10
    
    print(f"Go batch collision (100 tests, 10 placed): {go_time*1000:.2f} ms")
    print(f"Detected {np.sum(results_go)} collisions")
    print("=" * 60)
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Go Geometry Integration")
    print("=" * 60)
    print()
    
    all_passed = True
    
    all_passed &= test_point_in_polygon()
    all_passed &= test_segments_intersect()
    all_passed &= test_polygons_intersect()
    all_passed &= test_rotate_and_translate()
    all_passed &= test_bounding_box()
    all_passed &= test_batch_collision()
    
    benchmark_comparison()
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
