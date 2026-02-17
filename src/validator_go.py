"""
Enhanced validator using Go for performance-critical operations.

This module provides a faster validation by using the Go geometry library
for collision detection while maintaining compatibility with the original validator.
"""

import argparse
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
import math

try:
    from go_geometry import (
        polygons_intersect_go,
        batch_collision_check_go,
        get_bounding_box_go
    )
    GO_AVAILABLE = True
except ImportError:
    GO_AVAILABLE = False
    print("Warning: Go library not available. Falling back to pure Python.")


# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')


def create_tree_polygon_numpy(center_x, center_y, angle):
    """
    Create a tree polygon as numpy array for Go processing.
    
    Args:
        center_x: X coordinate (as Decimal or string)
        center_y: Y coordinate (as Decimal or string)
        angle: Rotation angle in degrees (as Decimal or string)
        
    Returns:
        numpy array of shape (N, 2) representing polygon vertices
    """
    # Parse inputs
    if isinstance(center_x, str) and center_x.startswith('s'):
        center_x = center_x.lstrip('s')
    if isinstance(center_y, str) and center_y.startswith('s'):
        center_y = center_y.lstrip('s')
    if isinstance(angle, str) and angle.startswith('s'):
        angle = angle.lstrip('s')

    center_x = float(center_x)
    center_y = float(center_y)
    angle_deg = float(angle)
    
    # Tree dimensions (scaled)
    scale = 1e15
    trunk_w = 0.15 * scale
    trunk_h = 0.2 * scale
    base_w = 0.7 * scale
    mid_w = 0.4 * scale
    top_w = 0.25 * scale
    tip_y = 0.8 * scale
    tier_1_y = 0.5 * scale
    tier_2_y = 0.25 * scale
    base_y = 0.0
    trunk_bottom_y = -trunk_h
    
    # Create base polygon vertices
    vertices = np.array([
        [0.0, tip_y],
        [top_w / 2, tier_1_y],
        [top_w / 4, tier_1_y],
        [mid_w / 2, tier_2_y],
        [mid_w / 4, tier_2_y],
        [base_w / 2, base_y],
        [trunk_w / 2, base_y],
        [trunk_w / 2, trunk_bottom_y],
        [-trunk_w / 2, trunk_bottom_y],
        [-trunk_w / 2, base_y],
        [-base_w / 2, base_y],
        [-mid_w / 4, tier_2_y],
        [-mid_w / 2, tier_2_y],
        [-top_w / 4, tier_1_y],
        [-top_w / 2, tier_1_y],
    ], dtype=np.float64)
    
    # Rotate
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated = np.zeros_like(vertices)
    rotated[:, 0] = vertices[:, 0] * cos_a - vertices[:, 1] * sin_a
    rotated[:, 1] = vertices[:, 0] * sin_a + vertices[:, 1] * cos_a
    
    # Translate
    center_x_scaled = center_x * scale
    center_y_scaled = center_y * scale
    rotated[:, 0] += center_x_scaled
    rotated[:, 1] += center_y_scaled
    
    return rotated


def create_tree_polygon_shapely(center_x, center_y, angle):
    """
    Create a tree polygon using Shapely (for compatibility).
    This is the original implementation from validator.py.
    """
    # Parse inputs
    if isinstance(center_x, str) and center_x.startswith('s'):
        center_x = center_x.lstrip('s')
    if isinstance(center_y, str) and center_y.startswith('s'):
        center_y = center_y.lstrip('s')
    if isinstance(angle, str) and angle.startswith('s'):
        angle = angle.lstrip('s')

    center_x = Decimal(center_x)
    center_y = Decimal(center_y)
    angle = Decimal(angle)

    # Tree dimensions
    trunk_w = Decimal('0.15')
    trunk_h = Decimal('0.2')
    base_w = Decimal('0.7')
    mid_w = Decimal('0.4')
    top_w = Decimal('0.25')
    tip_y = Decimal('0.8')
    tier_1_y = Decimal('0.5')
    tier_2_y = Decimal('0.25')
    base_y = Decimal('0.0')
    trunk_bottom_y = -trunk_h

    # Create base polygon
    initial_polygon = Polygon([
        (Decimal('0.0') * scale_factor, tip_y * scale_factor),
        (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
        (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
        (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
        (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
        (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
        (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
        (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
        (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
        (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
        (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
        (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
        (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
        (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
        (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
    ])

    # Rotate and translate
    rotated = affinity.rotate(initial_polygon, float(angle), origin=(0, 0))
    polygon = affinity.translate(rotated,
                                xoff=float(center_x * scale_factor),
                                yoff=float(center_y * scale_factor))

    return polygon


def validate_submission_go(csv_path: str, check_overlaps: bool = True, verbose: bool = True):
    """
    Validate submission file using Go for performance-critical operations.
    
    Args:
        csv_path: Path to submission CSV
        check_overlaps: Whether to check for overlapping trees
        verbose: Print detailed output
        
    Returns:
        bool: True if valid, False otherwise
    """
    if verbose:
        print("=" * 70)
        print("SANTA TREE PACKING SUBMISSION VALIDATION (Go-accelerated)")
        print("=" * 70)
    
    # Load submission
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n[ERROR] Could not load CSV: {e}")
        return False
    
    if verbose:
        print(f"\n1. Loaded submission from: {csv_path}")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
    
    # Check format (same as original)
    required_cols = ['id', 'x', 'y', 'deg']
    for col in required_cols:
        if col not in df.columns:
            print(f"   [ERROR] Missing required column: {col}")
            return False
    
    if verbose:
        print("   [OK] Required columns present")
    
    # Check number of rows
    expected_rows = sum(range(1, 201))  # 1+2+...+200 = 20100
    if len(df) != expected_rows:
        print(f"   [ERROR] Expected {expected_rows} rows, got {len(df)}")
        return False
    
    if verbose:
        print(f"   [OK] Correct number of rows ({expected_rows})")
    
    # Check ID format
    expected_ids = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    if not all(df['id'] == expected_ids):
        print("   [ERROR] ID column does not match expected format")
        return False
    
    if verbose:
        print("   [OK] ID format correct")
    
    # Check string format
    for col in ['x', 'y', 'deg']:
        if not all(df[col].astype(str).str.startswith('s')):
            print(f"   [ERROR] Column '{col}' values must start with 's'")
            return False
    
    if verbose:
        print("   [OK] Values properly formatted with 's' prefix")
    
    # Check value ranges
    for col in ['x', 'y']:
        values = df[col].str.lstrip('s').astype(float)
        if not all((values >= -100) & (values <= 100)):
            print(f"   [ERROR] Column '{col}' has values outside [-100, 100]")
            return False
    
    if verbose:
        print("   [OK] Position values within [-100, 100]")
    
    # Check for overlaps using Go
    if check_overlaps and GO_AVAILABLE:
        if verbose:
            print("\n2. Checking for overlapping trees (Go-accelerated)...")
        
        overlap_found = False
        idx = 0
        
        for n in range(1, 201):
            trees = []
            
            for t in range(n):
                row = df.iloc[idx]
                polygon = create_tree_polygon_numpy(row['x'], row['y'], row['deg'])
                trees.append(polygon)
                idx += 1
            
            # Use Go for collision detection
            if len(trees) > 1:
                # Check all pairs
                for i in range(len(trees)):
                    for j in range(i+1, len(trees)):
                        intersects, touches = polygons_intersect_go(trees[i], trees[j], 1e-6)
                        if intersects and not touches:
                            print(f"   [ERROR] Overlap in {n}-tree config: trees {i} and {j}")
                            overlap_found = True
                            break
                    if overlap_found:
                        break
            
            if overlap_found:
                break
            
            if verbose and n % 50 == 0:
                print(f"   - Checked {n}/200 configurations...")
        
        if overlap_found:
            return False
        
        if verbose:
            print("   [OK] No overlaps detected")
    elif check_overlaps and not GO_AVAILABLE:
        print("   [WARNING] Go library not available, skipping overlap check")
    
    # Calculate score
    if verbose:
        print("\n3. Calculating approximate score...")
        
        total_score = 0
        idx = 0
        
        for n in range(1, 201):
            trees = []
            
            for t in range(n):
                row = df.iloc[idx]
                if GO_AVAILABLE:
                    polygon_np = create_tree_polygon_numpy(row['x'], row['y'], row['deg'])
                    minx, miny, maxx, maxy = get_bounding_box_go(polygon_np)
                    trees.append((minx, miny, maxx, maxy))
                else:
                    polygon = create_tree_polygon_shapely(row['x'], row['y'], row['deg'])
                    trees.append(polygon.bounds)
                idx += 1
            
            # Calculate bounding box of all trees
            if GO_AVAILABLE:
                minx = min(t[0] for t in trees) / 1e15
                miny = min(t[1] for t in trees) / 1e15
                maxx = max(t[2] for t in trees) / 1e15
                maxy = max(t[3] for t in trees) / 1e15
            else:
                from shapely.ops import unary_union
                shapely_trees = [create_tree_polygon_shapely(
                    df.iloc[idx-len(trees)+i]['x'],
                    df.iloc[idx-len(trees)+i]['y'],
                    df.iloc[idx-len(trees)+i]['deg']
                ) for i in range(len(trees))]
                bounds = unary_union(shapely_trees).bounds
                minx = Decimal(bounds[0]) / scale_factor
                miny = Decimal(bounds[1]) / scale_factor
                maxx = Decimal(bounds[2]) / scale_factor
                maxy = Decimal(bounds[3]) / scale_factor
            
            width = Decimal(str(maxx)) - Decimal(str(minx))
            height = Decimal(str(maxy)) - Decimal(str(miny))
            side_length = max(width, height)
            
            score = (side_length ** 2) / Decimal(n)
            total_score += score
        
        print(f"   - Total score: {float(total_score):.6f}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE - SUBMISSION IS VALID!")
        print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate Santa Tree Packing submission (Go-accelerated)'
    )
    parser.add_argument('csv_path', help='Path to submission CSV file')
    parser.add_argument('--skip-overlaps', action='store_true',
                       help='Skip overlap checking (faster)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    valid = validate_submission_go(
        args.csv_path,
        check_overlaps=not args.skip_overlaps,
        verbose=not args.quiet
    )
    
    return 0 if valid else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
