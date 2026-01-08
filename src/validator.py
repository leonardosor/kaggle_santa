"""
Validate Santa Tree Packing Submission

Validates submission CSV format and checks for:
- Correct format (id, x, y, deg columns)
- All 20100 rows (sum of 1 to 200)
- String format with 's' prefix
- No overlapping trees
"""

import argparse
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree


# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')


def create_tree_polygon(center_x, center_y, angle):
    """
    Create a tree polygon with given position and rotation

    Args:
        center_x: X coordinate (as Decimal or string)
        center_y: Y coordinate (as Decimal or string)
        angle: Rotation angle in degrees (as Decimal or string)

    Returns:
        Shapely Polygon representing the tree
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


def validate_submission(csv_path: str, check_overlaps: bool = True, verbose: bool = True):
    """
    Validate submission file

    Args:
        csv_path: Path to submission CSV
        check_overlaps: Whether to check for overlapping trees (slow for large submissions)
        verbose: Print detailed output

    Returns:
        bool: True if valid, False otherwise
    """
    if verbose:
        print("=" * 70)
        print("SANTA TREE PACKING SUBMISSION VALIDATION")
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

    # Check format
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

    # Check string format (values prepended with 's')
    for col in ['x', 'y', 'deg']:
        if not all(df[col].astype(str).str.startswith('s')):
            print(f"   [ERROR] Column '{col}' values must start with 's'")
            return False

    if verbose:
        print("   [OK] Values properly formatted with 's' prefix")

    # Check value ranges (-100 to 100 for x, y)
    for col in ['x', 'y']:
        values = df[col].str.lstrip('s').astype(float)
        if not all((values >= -100) & (values <= 100)):
            print(f"   [ERROR] Column '{col}' has values outside [-100, 100]")
            return False

    if verbose:
        print("   [OK] Position values within [-100, 100]")

    # Check for overlaps (optional, slow)
    if check_overlaps:
        if verbose:
            print("\n2. Checking for overlapping trees...")

        overlap_found = False
        idx = 0

        for n in range(1, 201):
            trees = []

            for t in range(n):
                row = df.iloc[idx]
                polygon = create_tree_polygon(row['x'], row['y'], row['deg'])
                trees.append(polygon)
                idx += 1

            # Check for overlaps in this configuration
            tree_index = STRtree(trees)

            for i, tree in enumerate(trees):
                possible_overlaps = tree_index.query(tree)
                for j in possible_overlaps:
                    if i < j:  # Avoid checking same pair twice
                        if tree.intersects(trees[j]) and not tree.touches(trees[j]):
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

    # Calculate score (approximate)
    if verbose:
        print("\n3. Calculating approximate score...")

        total_score = 0
        idx = 0

        for n in range(1, 201):
            trees = []

            for t in range(n):
                row = df.iloc[idx]
                polygon = create_tree_polygon(row['x'], row['y'], row['deg'])
                trees.append(polygon)
                idx += 1

            # Calculate bounding box
            from shapely.ops import unary_union
            bounds = unary_union(trees).bounds

            minx = Decimal(bounds[0]) / scale_factor
            miny = Decimal(bounds[1]) / scale_factor
            maxx = Decimal(bounds[2]) / scale_factor
            maxy = Decimal(bounds[3]) / scale_factor

            width = maxx - minx
            height = maxy - miny
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
        description="Validate Santa 2025 submission file"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to submission CSV file"
    )
    parser.add_argument(
        "--no-overlap-check",
        action="store_true",
        help="Skip overlap checking (faster)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    is_valid = validate_submission(
        args.csv_path,
        check_overlaps=not args.no_overlap_check,
        verbose=not args.quiet
    )

    if not is_valid:
        exit(1)


if __name__ == "__main__":
    main()
