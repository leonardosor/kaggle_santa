"""Quick test of GPU optimizations with small workload."""
import sys
import time
from decimal import Decimal
import numpy as np


def quick_test():
    """Test GPU optimizations with first 10 trees."""
    print("=" * 70)
    print("GPU Optimization Quick Test")
    print("=" * 70)
    print()
    
    # Check CUDA availability
    try:
        import cupy as cp
        print(f"✓ CUDA available: {cp.cuda.runtime.getDeviceCount()} GPU(s)")
    except ImportError:
        print("✗ CUDA not available - will use CPU fallback")
    print()
    
    # Import after CUDA check
    from gpu_optimizations import initialize_trees_gpu
    from run_optimized import ChristmasTree
    
    print("Testing with 10 trees...")
    print("-" * 70)
    
    start_time = time.time()
    
    current_trees = []
    
    for n in range(1, 11):
        iter_start = time.time()
        
        current_trees, side = initialize_trees_gpu(
            n,
            existing_trees=current_trees,
            num_gpus=1,
            gpu_device=0,
            use_optimization=True
        )
        
        iter_time = time.time() - iter_start
        print(f"  {n} trees: side={side:.6f}, time={iter_time:.3f}s")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per configuration: {total_time/10:.3f}s")
    print("=" * 70)
    
    # Print adaptive performance stats
    try:
        from gpu_adaptive import print_performance_summary
        print_performance_summary()
    except:
        pass
    
    print()
    print("✓ Quick test completed successfully!")
    print()
    print("To run full optimization (200 trees):")
    print("  python src/run_gpu.py")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
