"""Run the optimized tree packing algorithm."""
import math
import random
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')

# Build the index of the submission
index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        if isinstance(center_x, str) and center_x.startswith('s'):
            center_x = center_x.lstrip('s')
        if isinstance(center_y, str) and center_y.startswith('s'):
            center_y = center_y.lstrip('s')
        if isinstance(angle, str) and angle.startswith('s'):
            angle = angle.lstrip('s')
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

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
        self.base_polygon = initial_polygon
        self._update_polygon()

    def _update_polygon(self):
        """Update the polygon based on current position and rotation."""
        rotated = affinity.rotate(self.base_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

    def set_position(self, x, y, angle=None):
        """Set tree position and optionally angle, updating the polygon."""
        self.center_x = x
        self.center_y = y
        if angle is not None:
            self.angle = angle
        self._update_polygon()

    def copy(self):
        """Create a deep copy of this tree."""
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))


def get_bounding_square(polygons):
    """Calculate the side length of the bounding square for a list of polygons."""
    if not polygons:
        return Decimal('0')

    bounds = unary_union(polygons).bounds
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    return max(width, height)


def check_collision(tree_poly, other_polygons, tree_index=None):
    """Check if a tree polygon collides with any other polygons (excluding touching)."""
    if tree_index is None:
        tree_index = STRtree(other_polygons) if other_polygons else None

    if not other_polygons:
        return False

    possible_indices = tree_index.query(tree_poly)
    return any((tree_poly.intersects(other_polygons[i]) and not tree_poly.touches(other_polygons[i]))
               for i in possible_indices)


def generate_weighted_angle():
    """Generate a random angle weighted by abs(sin(2*angle)) for better corner placement."""
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def place_tree_greedy(tree_to_place, placed_trees, num_attempts=15):
    """
    Place a tree using greedy approach with multiple random attempts.
    Returns the best position found.
    """
    if not placed_trees:
        tree_to_place.set_position(Decimal('0'), Decimal('0'))
        return tree_to_place

    placed_polygons = [p.polygon for p in placed_trees]
    tree_index = STRtree(placed_polygons)

    best_px = None
    best_py = None
    best_angle = tree_to_place.angle
    min_radius = Decimal('Infinity')

    for attempt in range(num_attempts):
        # Try different rotation angles for diversity
        test_angle = Decimal(random.uniform(0, 360)) if attempt > 0 else tree_to_place.angle
        temp_tree = tree_to_place.copy()
        temp_tree.set_position(Decimal('0'), Decimal('0'), test_angle)

        # Start far away
        angle = generate_weighted_angle()
        vx = Decimal(str(math.cos(angle)))
        vy = Decimal(str(math.sin(angle)))

        # Move towards center until collision
        radius = Decimal('25.0')
        step_in = Decimal('0.5')

        collision_found = False
        while radius >= 0:
            px = radius * vx
            py = radius * vy

            temp_tree.set_position(px, py)

            if check_collision(temp_tree.polygon, placed_polygons, tree_index):
                collision_found = True
                break
            radius -= step_in

        # Back up until no collision
        if collision_found:
            step_out = Decimal('0.05')
            while True:
                radius += step_out
                px = radius * vx
                py = radius * vy
                temp_tree.set_position(px, py)

                if not check_collision(temp_tree.polygon, placed_polygons, tree_index):
                    break
        else:
            radius = Decimal('0')
            px = Decimal('0')
            py = Decimal('0')

        if radius < min_radius:
            min_radius = radius
            best_px = px
            best_py = py
            best_angle = test_angle

    tree_to_place.set_position(best_px, best_py, best_angle)
    return tree_to_place


def optimize_tree_rotation(tree_idx, placed_trees, angle_step=5):
    """
    Optimize the rotation of a single tree to minimize bounding box.
    Returns True if improvement was found.
    """
    tree = placed_trees[tree_idx]
    other_trees = [t for i, t in enumerate(placed_trees) if i != tree_idx]
    other_polygons = [t.polygon for t in other_trees]
    tree_index = STRtree(other_polygons) if other_polygons else None

    current_polygons = [t.polygon for t in placed_trees]
    current_side = get_bounding_square(current_polygons)

    best_angle = tree.angle
    best_side = current_side
    improved = False

    for delta in range(angle_step, 360, angle_step):
        test_angle = tree.angle + Decimal(delta)
        temp_tree = tree.copy()
        temp_tree.set_position(tree.center_x, tree.center_y, test_angle)

        if not check_collision(temp_tree.polygon, other_polygons, tree_index):
            test_polygons = list(current_polygons)
            test_polygons[tree_idx] = temp_tree.polygon
            test_side = get_bounding_square(test_polygons)

            if test_side < best_side - Decimal('1e-10'):
                best_side = test_side
                best_angle = test_angle
                improved = True

    if improved:
        tree.set_position(tree.center_x, tree.center_y, best_angle)

    return improved


def optimize_tree_position(tree_idx, placed_trees, max_distance=0.5, step=0.05):
    """
    Try to move a tree slightly to reduce bounding box.
    Returns True if improvement was found.
    """
    tree = placed_trees[tree_idx]
    other_trees = [t for i, t in enumerate(placed_trees) if i != tree_idx]
    other_polygons = [t.polygon for t in other_trees]
    tree_index = STRtree(other_polygons) if other_polygons else None

    current_polygons = [t.polygon for t in placed_trees]
    current_side = get_bounding_square(current_polygons)

    best_x = tree.center_x
    best_y = tree.center_y
    best_side = current_side
    improved = False

    # Try moving in different directions
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(angle)
        dx = Decimal(str(math.cos(rad))) * Decimal(str(step))
        dy = Decimal(str(math.sin(rad))) * Decimal(str(step))

        distance = Decimal('0')
        while distance < Decimal(str(max_distance)):
            distance += Decimal(str(step))
            test_x = tree.center_x + dx * (distance / Decimal(str(step)))
            test_y = tree.center_y + dy * (distance / Decimal(str(step)))

            temp_tree = tree.copy()
            temp_tree.set_position(test_x, test_y, tree.angle)

            if not check_collision(temp_tree.polygon, other_polygons, tree_index):
                test_polygons = list(current_polygons)
                test_polygons[tree_idx] = temp_tree.polygon
                test_side = get_bounding_square(test_polygons)

                if test_side < best_side - Decimal('1e-10'):
                    best_side = test_side
                    best_x = test_x
                    best_y = test_y
                    improved = True
            else:
                break

    if improved:
        tree.set_position(best_x, best_y)

    return improved


def local_search_optimization(placed_trees, max_iterations=50):
    """
    Perform local search by trying to rotate and move each tree.
    """
    num_trees = len(placed_trees)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Try to optimize each tree
        for idx in range(num_trees):
            # First try rotation
            if optimize_tree_rotation(idx, placed_trees, angle_step=5):
                improved = True

            # Then try position adjustment
            if optimize_tree_position(idx, placed_trees, max_distance=0.3, step=0.05):
                improved = True

        if iteration % 10 == 0:
            current_side = get_bounding_square([t.polygon for t in placed_trees])
            print(f"  Local search iteration {iteration}: side = {current_side:.6f}")

    return placed_trees


def simulated_annealing(placed_trees, initial_temp=1.0, cooling_rate=0.95, max_iterations=100):
    """
    Apply simulated annealing to escape local minima.
    """
    current_trees = [t.copy() for t in placed_trees]
    current_side = get_bounding_square([t.polygon for t in current_trees])

    best_trees = [t.copy() for t in current_trees]
    best_side = current_side

    temp = initial_temp
    num_trees = len(placed_trees)

    for iteration in range(max_iterations):
        # Randomly perturb a tree
        tree_idx = random.randint(0, num_trees - 1)
        tree = current_trees[tree_idx]

        # Save original state
        orig_x, orig_y, orig_angle = tree.center_x, tree.center_y, tree.angle

        # Random perturbation
        if random.random() < 0.5:
            # Perturb position
            delta_x = Decimal(str(random.uniform(-0.5, 0.5))) * Decimal(str(temp))
            delta_y = Decimal(str(random.uniform(-0.5, 0.5))) * Decimal(str(temp))
            new_x = orig_x + delta_x
            new_y = orig_y + delta_y
            tree.set_position(new_x, new_y)
        else:
            # Perturb angle
            delta_angle = Decimal(str(random.uniform(-30, 30))) * Decimal(str(temp))
            new_angle = orig_angle + delta_angle
            tree.set_position(orig_x, orig_y, new_angle)

        # Check collision
        other_trees = [t for i, t in enumerate(current_trees) if i != tree_idx]
        other_polygons = [t.polygon for t in other_trees]

        if check_collision(tree.polygon, other_polygons):
            # Revert if collision
            tree.set_position(orig_x, orig_y, orig_angle)
        else:
            # Evaluate new configuration
            new_side = get_bounding_square([t.polygon for t in current_trees])
            delta_side = float(new_side - current_side)

            # Accept or reject based on Metropolis criterion
            if delta_side < 0 or random.random() < math.exp(-delta_side / temp):
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_trees = [t.copy() for t in current_trees]
            else:
                # Revert
                tree.set_position(orig_x, orig_y, orig_angle)

        # Cool down
        if (iteration + 1) % 20 == 0:
            temp *= cooling_rate
            print(f"  SA iteration {iteration + 1}: temp = {temp:.4f}, best = {best_side:.6f}")

    return best_trees


def initialize_trees(num_trees, existing_trees=None, use_optimization=True):
    """
    Initialize trees with improved placement and optional optimization.
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = [t.copy() for t in existing_trees]

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        unplaced_trees = [
            ChristmasTree(angle=random.uniform(0, 360)) for _ in range(num_to_add)
        ]

        # Place first tree at origin if starting from scratch
        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        # Place remaining trees greedily
        for tree_to_place in unplaced_trees:
            place_tree_greedy(tree_to_place, placed_trees, num_attempts=20)
            placed_trees.append(tree_to_place)

    # Apply optimizations for larger configurations
    if use_optimization:
        if num_trees >= 5:
            print(f"  Running local search optimization...")
            placed_trees = local_search_optimization(placed_trees, max_iterations=30)

        if num_trees >= 10 and num_trees % 10 == 0:
            print(f"  Running simulated annealing...")
            placed_trees = simulated_annealing(placed_trees,
                                              initial_temp=0.5,
                                              cooling_rate=0.95,
                                              max_iterations=50)

    all_polygons = [t.polygon for t in placed_trees]
    side_length = get_bounding_square(all_polygons)

    return placed_trees, side_length


def main():
    """Main optimization loop."""
    tree_data = []
    current_placed_trees = []
    scores = []

    print("Starting optimized tree packing algorithm...")
    print("=" * 60)

    for n in range(200):
        print(f"\nProcessing {n+1} trees...")

        current_placed_trees, side = initialize_trees(
            n+1,
            existing_trees=current_placed_trees,
            use_optimization=True
        )

        print(f"Final side length for {n+1} trees: {side:.12f}")
        scores.append(float(side))

        # Store data for all trees in current configuration
        current_tree_data = []
        for tree in current_placed_trees:
            current_tree_data.append([tree.center_x, tree.center_y, tree.angle])

        # Append all trees from this configuration
        tree_data.extend(current_tree_data)

    # Create submission
    print("\n" + "=" * 60)
    print("Creating submission file...")

    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=tree_data
    ).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    # Prepend 's' to keep as string
    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')

    output_path = Path(__file__).parent.parent / 'output' / 'submission_optimized.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)

    print(f"Submission saved to {output_path}")
    print(f"Total configurations: {len(submission)}")

    # Save scores for comparison
    scores_df = pd.DataFrame({
        'num_trees': range(1, 201),
        'side_length': scores
    })
    scores_path = Path(__file__).parent.parent / 'output' / 'scores_optimized.csv'
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}")

    # Calculate total score (sum of all side lengths)
    total_score = sum(scores)
    print(f"\nTotal Score: {total_score:.6f}")

    return total_score


if __name__ == "__main__":
    main()
