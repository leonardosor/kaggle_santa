"""Run enhanced tree packing algorithm with all improvements."""
import sys
import os
import random
from pathlib import Path

# Redirect output
log_path = Path(__file__).parent.parent / 'output' / 'enhanced_run.log'
sys.stdout = open(log_path, 'w', buffering=1)
sys.stderr = sys.stdout

print("Starting ENHANCED tree packing algorithm...", flush=True)
print("=" * 60, flush=True)
print("Strategies: Multi-start, Corner-first, Polar, Adaptive steps", flush=True)
print("=" * 60, flush=True)

from optimizations_enhanced import initialize_trees_enhanced
from run_optimized import index, scale_factor
from decimal import Decimal
import pandas as pd


def main():
    """Main enhanced optimization loop."""
    tree_data = []
    current_placed_trees = []
    scores = []

    for n in range(200):
        print(f"\nProcessing {n+1} trees...", flush=True)

        # Choose strategy based on tree count
        if n + 1 <= 10:
            strategy = 'corner_first'
        elif n + 1 <= 30:
            strategy = 'polar'
        else:
            strategy = 'hybrid'

        current_placed_trees, side = initialize_trees_enhanced(
            n + 1,
            existing_trees=current_placed_trees,
            strategy=strategy,
            use_optimization=True
        )

        print(f"Strategy: {strategy}, Side length: {side:.12f}", flush=True)
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

    output_path = Path(__file__).parent.parent / 'output' / 'submission_enhanced.csv'
    submission.to_csv(output_path)

    print(f"Submission saved to {output_path}", flush=True)

    # Save scores
    scores_df = pd.DataFrame({
        'num_trees': range(1, 201),
        'side_length': scores
    })
    scores_path = Path(__file__).parent.parent / 'output' / 'scores_enhanced.csv'
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}", flush=True)

    # Calculate total score
    total_score = sum(scores)
    print(f"\nTotal Score: {total_score:.6f}", flush=True)

    sys.stdout.close()
    return total_score


if __name__ == "__main__":
    main()
