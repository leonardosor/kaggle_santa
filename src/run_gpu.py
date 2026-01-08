"""Run GPU-accelerated tree packing algorithm."""
import sys
import os
from pathlib import Path
from datetime import datetime


class TeeOutput:
    """Write to both terminal and log file simultaneously."""
    def __init__(self, log_path, original_stdout):
        self.terminal = original_stdout
        self.log = open(log_path, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Setup tee output to both terminal and log file
# Use timestamp to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = Path(__file__).parent.parent / 'output' / f'gpu_run_{timestamp}.log'
original_stdout = sys.stdout
sys.stdout = TeeOutput(log_path, original_stdout)
sys.stderr = sys.stdout

print("=" * 80, flush=True)
print("GPU-ACCELERATED TREE PACKING ALGORITHM", flush=True)
print(f"Run timestamp: {timestamp}", flush=True)
print("=" * 80, flush=True)

try:
    import cupy as cp
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Detected {num_gpus} CUDA GPUs", flush=True)

    # Find GPU with most memory
    max_memory = 0
    primary_gpu = 0

    for i in range(num_gpus):
        with cp.cuda.Device(i):
            props = cp.cuda.runtime.getDeviceProperties(i)
            total_memory = props['totalGlobalMem']
            memory_gb = total_memory / (1024**3)
            print(f"  GPU {i}: {props['name'].decode()} ({memory_gb:.2f} GB)", flush=True)

            if total_memory > max_memory:
                max_memory = total_memory
                primary_gpu = i

    if num_gpus > 0:
        print(f"\nUsing GPU {primary_gpu} as primary (most memory: {max_memory/(1024**3):.2f} GB)", flush=True)
        cp.cuda.Device(primary_gpu).use()

except ImportError:
    print("WARNING: CuPy not installed. Falling back to CPU.", flush=True)
    print("To install CuPy:", flush=True)
    print("  - For CUDA 11.x: pip install cupy-cuda11x", flush=True)
    print("  - For CUDA 12.x: pip install cupy-cuda12x", flush=True)
    num_gpus = 0
    primary_gpu = 0

print("=" * 80, flush=True)

from gpu_optimizations import initialize_trees_gpu
from run_optimized import index
from decimal import Decimal
import pandas as pd
import time


def main():
    """Main GPU-accelerated optimization loop."""
    global original_stdout

    tree_data = []
    current_placed_trees = []
    scores = []

    # Use only the primary GPU (the one with most memory)
    use_gpus = 1
    gpu_device = primary_gpu if num_gpus > 0 else 0

    start_time = time.time()

    for n in range(200):
        iter_start = time.time()
        print(f"\nProcessing {n+1} trees...", flush=True)

        current_placed_trees, side = initialize_trees_gpu(
            n + 1,
            existing_trees=current_placed_trees,
            num_gpus=use_gpus,
            gpu_device=gpu_device,
            use_optimization=True
        )

        iter_time = time.time() - iter_start
        print(f"Side length: {side:.12f} (took {iter_time:.2f}s)", flush=True)
        scores.append(float(side))

        # Store data for all trees in current configuration
        for tree in current_placed_trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])

    total_time = time.time() - start_time

    # Create submission
    print("\n" + "=" * 80, flush=True)
    print("Creating submission file...", flush=True)

    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=tree_data
    ).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')

    # Use timestamp to avoid overwriting previous runs
    output_path = Path(__file__).parent.parent / 'output' / f'submission_gpu_{timestamp}.csv'
    submission.to_csv(output_path)

    print(f"Submission saved to {output_path}", flush=True)

    # Save scores
    scores_df = pd.DataFrame({
        'num_trees': range(1, 201),
        'side_length': scores
    })
    scores_path = Path(__file__).parent.parent / 'output' / f'scores_gpu_{timestamp}.csv'
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}", flush=True)
    
    # Also save as latest for easy access
    latest_submission = Path(__file__).parent.parent / 'output' / 'submission_gpu_latest.csv'
    submission.to_csv(latest_submission)
    latest_scores = Path(__file__).parent.parent / 'output' / 'scores_gpu_latest.csv'
    scores_df.to_csv(latest_scores, index=False)
    print(f"Latest versions also saved (submission_gpu_latest.csv, scores_gpu_latest.csv)", flush=True)

    # Calculate total score
    total_score = sum(scores)
    print(f"\n" + "=" * 80, flush=True)
    print(f"FINAL RESULTS", flush=True)
    print(f"=" * 80, flush=True)
    print(f"Total Score: {total_score:.6f}", flush=True)
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)", flush=True)
    print(f"Average time per configuration: {total_time/200:.2f}s", flush=True)
    print(f"=" * 80, flush=True)

    # Close log file (restore original stdout)
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = original_stdout

    return total_score


if __name__ == "__main__":
    main()
