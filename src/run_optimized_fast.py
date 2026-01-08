"""Run optimized algorithm with reduced optimization intensity for faster completion."""
import sys
import os

# Redirect stdout and stderr immediately
sys.stdout = open(os.path.join('..', 'output', 'optimized_fast_run.log'), 'w', buffering=1)
sys.stderr = sys.stdout

print("Starting FAST optimized tree packing algorithm...", flush=True)
print("=" * 60, flush=True)

# Now import everything else
from run_optimized import (
    ChristmasTree, get_bounding_square, check_collision,
    generate_weighted_angle, place_tree_greedy,
    optimize_tree_rotation, optimize_tree_position,
    local_search_optimization, simulated_annealing,
    index, scale_factor
)
from decimal import Decimal
import pandas as pd
from pathlib import Path

def initialize_trees_fast(num_trees, existing_trees=None):
    """Initialize trees with FAST optimization (reduced intensity)."""
    print(f"\nProcessing {num_trees} trees...", flush=True)

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

        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            place_tree_greedy(tree_to_place, placed_trees, num_attempts=15)  # Reduced from 20
            placed_trees.append(tree_to_place)

    # FAST optimization: Less intensive
    if num_trees >= 10:
        print(f"  Running quick local search...", flush=True)
        # Quick local search - only 10 iterations max
        num_trees_count = len(placed_trees)
        iteration = 0
        improved = True

        while improved and iteration < 10:  # Reduced from 30
            improved = False
            iteration += 1

            for idx in range(num_trees_count):
                if optimize_tree_rotation(idx, placed_trees, angle_step=10):  # Faster steps
                    improved = True

        current_side = get_bounding_square([t.polygon for t in placed_trees])
        print(f"  After optimization: side = {current_side:.6f}", flush=True)

    # Skip simulated annealing for speed

    all_polygons = [t.polygon for t in placed_trees]
    side_length = get_bounding_square(all_polygons)

    print(f"Final side length for {num_trees} trees: {side_length:.12f}", flush=True)
    return placed_trees, side_length


import random

def main():
    """Main fast optimization loop."""
    tree_data = []
    current_placed_trees = []
    scores = []

    for n in range(200):
        current_placed_trees, side = initialize_trees_fast(
            n+1,
            existing_trees=current_placed_trees
        )

        scores.append(float(side))

        # Store data for all trees in current configuration
        for tree in current_placed_trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])

    # Create submission
    print("\n" + "=" * 60, flush=True)
    print("Creating submission file...", flush=True)

    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=tree_data
    ).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')

    output_path = Path(__file__).parent.parent / 'output' / 'submission_optimized_fast.csv'
    submission.to_csv(output_path)

    print(f"Submission saved to {output_path}", flush=True)

    # Save scores
    scores_df = pd.DataFrame({
        'num_trees': range(1, 201),
        'side_length': scores
    })
    scores_path = Path(__file__).parent.parent / 'output' / 'scores_optimized_fast.csv'
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}", flush=True)

    # Calculate total score
    total_score = sum(scores)
    print(f"\nTotal Score: {total_score:.6f}", flush=True)

    sys.stdout.close()
    return total_score


if __name__ == "__main__":
    main()
