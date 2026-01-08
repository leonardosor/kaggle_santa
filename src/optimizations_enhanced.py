"""Enhanced optimization strategies for tree packing."""
import math
import random
from decimal import Decimal

from run_optimized import (
    ChristmasTree, get_bounding_square, check_collision,
    place_tree_greedy, optimize_tree_rotation, optimize_tree_position,
    scale_factor
)
from shapely import affinity
from shapely.strtree import STRtree


# ============================================================================
# STRATEGY 1: Multi-Start Optimization
# ============================================================================

def multi_start_placement(num_to_place, placed_trees, num_starts=3):
    """
    Try placing a tree multiple times with different seeds and keep the best.
    """
    best_tree = None
    best_radius = Decimal('Infinity')

    for seed_offset in range(num_starts):
        # Create a new tree with different random angle
        test_tree = ChristmasTree(angle=random.uniform(0, 360))

        # Use the enhanced greedy placement
        place_tree_greedy(test_tree, placed_trees, num_attempts=15)

        # Calculate distance from origin as heuristic
        radius = (test_tree.center_x ** 2 + test_tree.center_y ** 2).sqrt()

        if radius < best_radius:
            best_radius = radius
            best_tree = test_tree.copy()

    return best_tree


# ============================================================================
# STRATEGY 2: Corner-First Placement
# ============================================================================

def estimate_corner_angle(x_sign, y_sign):
    """Estimate good angle for corner placement based on tree shape."""
    # Trees are wider at bottom, so angle them toward corners
    base_angle = math.degrees(math.atan2(float(y_sign), float(x_sign)))
    # Add 45 degrees to align with diagonal
    return (base_angle + 45) % 360


def corner_first_placement(num_trees, placed_trees, current_idx):
    """
    For the first few trees, use corner-first strategy.
    Returns a tree positioned strategically for corners.
    """
    if current_idx >= 4:
        return None  # Only apply to first 4 trees

    # Define corner positions (unit vectors toward corners)
    corners = [
        (Decimal('1'), Decimal('1')),      # Top-right
        (Decimal('-1'), Decimal('1')),     # Top-left
        (Decimal('1'), Decimal('-1')),     # Bottom-right
        (Decimal('-1'), Decimal('-1'))     # Bottom-left
    ]

    x_sign, y_sign = corners[current_idx]

    # Start with a reasonable distance from origin
    distance = Decimal('2.0')
    angle = estimate_corner_angle(x_sign, y_sign)

    tree = ChristmasTree(
        center_x=str(x_sign * distance),
        center_y=str(y_sign * distance),
        angle=str(angle)
    )

    # Check if this position is valid
    if placed_trees:
        placed_polygons = [p.polygon for p in placed_trees]
        tree_index = STRtree(placed_polygons)

        # If collision, fall back to greedy
        if check_collision(tree.polygon, placed_polygons, tree_index):
            return None

    return tree


# ============================================================================
# STRATEGY 3: Polar Coordinate Initialization
# ============================================================================

def polar_coordinate_placement(tree_idx, num_trees, placed_trees):
    """
    Place trees in a polar coordinate pattern for more uniform distribution.
    """
    # Use golden angle for optimal distribution
    golden_angle = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees

    angle = golden_angle * tree_idx

    # Radius increases with square root for area-proportional spacing
    radius = Decimal(str(math.sqrt(tree_idx + 1) * 1.5))

    x = radius * Decimal(str(math.cos(angle)))
    y = radius * Decimal(str(math.sin(angle)))

    # Random rotation for diversity
    rotation = Decimal(str(random.uniform(0, 360)))

    tree = ChristmasTree(str(x), str(y), str(rotation))

    # Check collision
    if placed_trees:
        placed_polygons = [p.polygon for p in placed_trees]
        if check_collision(tree.polygon, placed_polygons):
            return None

    return tree


# ============================================================================
# STRATEGY 4: Adaptive Step Sizes
# ============================================================================

def adaptive_local_search(placed_trees, initial_step=0.3, min_step=0.01, max_iterations=50):
    """
    Perform local search with adaptive step sizes.
    """
    step_size = Decimal(str(initial_step))
    min_step_size = Decimal(str(min_step))
    num_trees = len(placed_trees)

    iteration = 0
    no_improvement_count = 0

    while iteration < max_iterations and step_size >= min_step_size:
        improved = False
        iteration += 1

        # Try to optimize each tree
        for idx in range(num_trees):
            # Rotation optimization
            if optimize_tree_rotation(idx, placed_trees, angle_step=5):
                improved = True
                no_improvement_count = 0

            # Position optimization with current step size
            if optimize_tree_position(idx, placed_trees,
                                     max_distance=float(step_size),
                                     step=float(step_size / 5)):
                improved = True
                no_improvement_count = 0

        if not improved:
            no_improvement_count += 1

            # Reduce step size after several iterations without improvement
            if no_improvement_count >= 3:
                step_size *= Decimal('0.7')
                no_improvement_count = 0
                if step_size < min_step_size:
                    break

    return placed_trees


# ============================================================================
# STRATEGY 5: Intelligent Tree Selection Order
# ============================================================================

def compute_tree_priority(tree, placed_trees):
    """
    Compute priority for optimizing a tree.
    Trees further from center or on border get higher priority.
    """
    # Distance from origin
    distance = float((tree.center_x ** 2 + tree.center_y ** 2).sqrt())

    # Check if on border (no trees nearby in some direction)
    all_polygons = [t.polygon for t in placed_trees]
    bounds = get_bounding_square(all_polygons)

    # Trees closer to bounding box edge get higher priority
    edge_distance = float(bounds) / 2 - distance

    return distance + (1.0 / (edge_distance + 0.1))


def priority_based_optimization(placed_trees, max_iterations=30):
    """
    Optimize trees in priority order (border trees first).
    """
    num_trees = len(placed_trees)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Compute priorities
        priorities = [
            (compute_tree_priority(placed_trees[i], placed_trees), i)
            for i in range(num_trees)
        ]

        # Sort by priority (descending)
        priorities.sort(reverse=True)

        # Optimize in priority order
        for _, idx in priorities[:min(10, num_trees)]:  # Top 10 priority trees
            if optimize_tree_rotation(idx, placed_trees, angle_step=5):
                improved = True

            if optimize_tree_position(idx, placed_trees, max_distance=0.2, step=0.04):
                improved = True

    return placed_trees


# ============================================================================
# STRATEGY 6: Spiral Placement Pattern
# ============================================================================

def spiral_placement(tree_idx, num_trees):
    """
    Place trees in an Archimedean spiral pattern.
    """
    # Spiral parameters
    a = 0.5  # Spiral tightness
    b = 0.3  # Spacing between spiral arms

    theta = math.sqrt(tree_idx * 4 * math.pi)
    radius = a + b * theta

    x = Decimal(str(radius * math.cos(theta)))
    y = Decimal(str(radius * math.sin(theta)))

    # Rotation aligned with spiral tangent
    rotation = Decimal(str((math.degrees(theta) + 90) % 360))

    return ChristmasTree(str(x), str(y), str(rotation))


# ============================================================================
# STRATEGY 7: Density-Based Refinement
# ============================================================================

def compute_local_density(tree_idx, placed_trees, radius=2.0):
    """
    Compute how many trees are within a certain radius of the given tree.
    """
    tree = placed_trees[tree_idx]
    count = 0
    radius_sq = Decimal(str(radius)) ** 2

    for i, other in enumerate(placed_trees):
        if i == tree_idx:
            continue

        dist_sq = (tree.center_x - other.center_x) ** 2 + \
                  (tree.center_y - other.center_y) ** 2

        if dist_sq < radius_sq:
            count += 1

    return count


def density_aware_optimization(placed_trees, max_iterations=20):
    """
    Focus optimization on high-density regions.
    """
    num_trees = len(placed_trees)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Compute density for each tree
        densities = [
            (compute_local_density(i, placed_trees), i)
            for i in range(num_trees)
        ]

        # Sort by density (descending - optimize crowded areas first)
        densities.sort(reverse=True)

        # Optimize high-density trees
        for _, idx in densities[:min(15, num_trees)]:
            if optimize_tree_rotation(idx, placed_trees, angle_step=3):
                improved = True

            if optimize_tree_position(idx, placed_trees, max_distance=0.15, step=0.03):
                improved = True

    return placed_trees


# ============================================================================
# COMBINED ENHANCED INITIALIZATION
# ============================================================================

def initialize_trees_enhanced(num_trees, existing_trees=None,
                              strategy='hybrid', use_optimization=True):
    """
    Enhanced initialization combining multiple strategies.

    Strategies:
        - 'corner_first': Start with corners, then greedy
        - 'polar': Polar coordinate initialization
        - 'spiral': Spiral pattern initialization
        - 'hybrid': Mix of strategies based on tree count
        - 'multi_start': Multiple attempts with best selection
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = [t.copy() for t in existing_trees]

    num_to_add = num_trees - len(placed_trees)
    start_idx = len(placed_trees)

    if num_to_add > 0:
        # Place first tree at origin if starting from scratch
        if not placed_trees:
            first_tree = ChristmasTree(angle=str(random.uniform(0, 360)))
            placed_trees.append(first_tree)
            num_to_add -= 1
            start_idx = 1

        # Place remaining trees based on strategy
        for i in range(num_to_add):
            tree_idx = start_idx + i
            tree = None

            if strategy == 'corner_first' and tree_idx < 4:
                tree = corner_first_placement(num_trees, placed_trees, tree_idx)

            elif strategy == 'polar':
                tree = polar_coordinate_placement(tree_idx, num_trees, placed_trees)

            elif strategy == 'spiral':
                tree = spiral_placement(tree_idx, num_trees)
                # Validate no collision
                if placed_trees:
                    placed_polygons = [p.polygon for p in placed_trees]
                    if check_collision(tree.polygon, placed_polygons):
                        tree = None

            elif strategy == 'hybrid':
                # Use corner-first for first 4, then greedy with multi-start
                if tree_idx < 4:
                    tree = corner_first_placement(num_trees, placed_trees, tree_idx)
                elif tree_idx < 20:
                    tree = polar_coordinate_placement(tree_idx, num_trees, placed_trees)

            elif strategy == 'multi_start':
                tree = multi_start_placement(1, placed_trees, num_starts=3)

            # Fallback to greedy if strategy didn't produce a valid tree
            if tree is None:
                tree = ChristmasTree(angle=random.uniform(0, 360))
                place_tree_greedy(tree, placed_trees, num_attempts=20)

            placed_trees.append(tree)

    # Apply optimizations
    if use_optimization and num_trees >= 5:
        if num_trees < 20:
            # Light optimization for small configs
            placed_trees = adaptive_local_search(placed_trees,
                                                initial_step=0.3,
                                                max_iterations=20)
        elif num_trees < 50:
            # Medium optimization
            placed_trees = priority_based_optimization(placed_trees, max_iterations=25)
            placed_trees = adaptive_local_search(placed_trees,
                                                initial_step=0.25,
                                                max_iterations=30)
        else:
            # Heavy optimization for large configs
            placed_trees = density_aware_optimization(placed_trees, max_iterations=20)
            placed_trees = priority_based_optimization(placed_trees, max_iterations=30)
            placed_trees = adaptive_local_search(placed_trees,
                                                initial_step=0.2,
                                                max_iterations=40)

    all_polygons = [t.polygon for t in placed_trees]
    side_length = get_bounding_square(all_polygons)

    return placed_trees, side_length
