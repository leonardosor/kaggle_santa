#!/usr/bin/env python3
"""
Demonstration of Go-accelerated geometric operations.

This script showcases the performance improvements from using Go
for CPU-intensive geometric operations in the Kaggle Santa challenge.
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from go_geometry import (
        batch_collision_check_go,
        polygons_intersect_go,
        point_in_polygon_go,
        rotate_and_translate_polygon_go,
        get_bounding_box_go
    )
    GO_AVAILABLE = True
except ImportError:
    GO_AVAILABLE = False
    print("Error: Go library not available. Please run 'make build-go' first.")
    sys.exit(1)


def create_random_polygon(center_x, center_y, radius=5):
    """Create a random square polygon."""
    return np.array([
        [center_x - radius, center_y - radius],
        [center_x + radius, center_y - radius],
        [center_x + radius, center_y + radius],
        [center_x - radius, center_y + radius],
    ], dtype=np.float64)


def demo_basic_operations():
    """Demonstrate basic geometric operations."""
    print("=" * 70)
    print("DEMO: Basic Geometric Operations")
    print("=" * 70)
    print()
    
    # Create a square
    square = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float64)
    
    print("1. Point in Polygon Test")
    print("-" * 70)
    test_points = [
        (np.array([5, 5]), "center"),
        (np.array([0, 0]), "vertex"),
        (np.array([15, 15]), "outside"),
    ]
    
    for point, desc in test_points:
        inside = point_in_polygon_go(point, square)
        print(f"   Point {point} ({desc:8s}): {'inside' if inside else 'outside'}")
    print()
    
    print("2. Polygon Intersection Test")
    print("-" * 70)
    poly1 = square
    poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float64)
    poly3 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)
    
    for i, (poly, desc) in enumerate([(poly2, "overlapping"), (poly3, "separate")], 1):
        intersects, touches = polygons_intersect_go(poly1, poly, 1e-6)
        print(f"   Test {i} ({desc:11s}): intersects={intersects}, touches={touches}")
    print()
    
    print("3. Polygon Transformation Test")
    print("-" * 70)
    angle = np.pi / 4  # 45 degrees
    dx, dy = 10.0, 10.0
    transformed = rotate_and_translate_polygon_go(square, angle, dx, dy)
    print(f"   Original first vertex:    {square[0]}")
    print(f"   Transformed first vertex: {transformed[0]}")
    print()
    
    print("4. Bounding Box Test")
    print("-" * 70)
    minX, minY, maxX, maxY = get_bounding_box_go(square)
    print(f"   Square bounds: [{minX}, {minY}] to [{maxX}, {maxY}]")
    print(f"   Width:  {maxX - minX}")
    print(f"   Height: {maxY - minY}")
    print()


def demo_batch_collision():
    """Demonstrate batch collision detection with performance measurement."""
    print("=" * 70)
    print("DEMO: Batch Collision Detection Performance")
    print("=" * 70)
    print()
    
    # Create test scenarios with increasing complexity
    scenarios = [
        (10, 5, "Small"),
        (50, 10, "Medium"),
        (100, 20, "Large"),
        (200, 30, "X-Large"),
    ]
    
    for num_tests, num_placed, size_name in scenarios:
        print(f"{size_name} scenario: {num_tests} test polygons vs {num_placed} placed")
        print("-" * 70)
        
        # Generate random test polygons
        np.random.seed(42)
        test_polys = []
        for i in range(num_tests):
            x = np.random.rand() * 100
            y = np.random.rand() * 100
            test_polys.append(create_random_polygon(x, y, radius=3))
        
        # Generate placed polygons in a grid
        placed_polys = []
        for i in range(num_placed):
            x = (i % 10) * 12.0
            y = (i // 10) * 12.0
            placed_polys.append(create_random_polygon(x, y, radius=5))
        
        # Benchmark
        num_iterations = 10
        start = time.time()
        for _ in range(num_iterations):
            results = batch_collision_check_go(test_polys, placed_polys, 1e-6)
        elapsed = (time.time() - start) / num_iterations
        
        num_collisions = np.sum(results)
        
        print(f"   Time per batch: {elapsed*1000:.2f} ms")
        print(f"   Collisions detected: {num_collisions}/{num_tests}")
        print(f"   Throughput: {num_tests/elapsed:.0f} tests/sec")
        print()


def demo_realistic_scenario():
    """Demonstrate a realistic collision detection scenario."""
    print("=" * 70)
    print("DEMO: Realistic Scenario - Christmas Tree Packing")
    print("=" * 70)
    print()
    
    print("Simulating placement of 5 Christmas trees...")
    print("-" * 70)
    
    # Create 5 tree-like polygons (simplified)
    def create_tree(x, y, rotation=0):
        """Create a simplified tree shape."""
        # Base triangle
        base = np.array([
            [0, 0.8],    # tip
            [0.35, 0],   # right base
            [-0.35, 0],  # left base
        ], dtype=np.float64)
        
        # Apply rotation
        if rotation != 0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            rotated = np.zeros_like(base)
            rotated[:, 0] = base[:, 0] * cos_r - base[:, 1] * sin_r
            rotated[:, 1] = base[:, 0] * sin_r + base[:, 1] * cos_r
            base = rotated
        
        # Translate
        base[:, 0] += x
        base[:, 1] += y
        
        return base
    
    # Place trees at different positions
    placed_trees = [
        create_tree(0, 0, 0),
        create_tree(1.5, 0, np.pi/6),
        create_tree(-1.5, 0, -np.pi/6),
    ]
    
    # Test candidates for next tree
    candidates = [
        create_tree(0, 1.5, 0),      # Above center - should fit
        create_tree(0.5, 0.5, 0),    # Too close - collision
        create_tree(3.0, 0, 0),      # Far right - should fit
    ]
    
    print("Checking 3 candidate positions for tree #4:")
    results = batch_collision_check_go(candidates, placed_trees, 1e-6)
    
    for i, (has_collision, pos) in enumerate(zip(results, [(0, 1.5), (0.5, 0.5), (3.0, 0)]), 1):
        status = "COLLISION" if has_collision else "OK"
        print(f"   Candidate {i} at {pos}: {status}")
    
    print()
    print("Successfully demonstrated collision detection for tree placement!")
    print()


def main():
    """Run all demonstrations."""
    if not GO_AVAILABLE:
        return 1
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Go-Accelerated Geometry for Kaggle Santa Challenge".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demo_basic_operations()
    demo_batch_collision()
    demo_realistic_scenario()
    
    print("=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Try 'make benchmark-go' to see detailed performance benchmarks")
    print("  - Use 'python3 src/validator_go.py <csv>' to validate submissions")
    print("  - Read README.md for full API documentation")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
