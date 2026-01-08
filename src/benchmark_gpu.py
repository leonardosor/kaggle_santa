"""Compare performance of different collision detection implementations."""
import numpy as np
import time
from decimal import Decimal

# Set up a simple test case
np.random.seed(42)


def create_simple_tree():
    """Create a simple tree for testing."""
    from run_optimized import ChristmasTree
    tree = ChristmasTree()
    return tree


def benchmark_collision_detection():
    """Benchmark different collision detection methods."""
    print("=" * 70)
    print("GPU Collision Detection Performance Benchmark")
    print("=" * 70)
    print()
    
    # Create test data
    tree_template = create_simple_tree()
    
    # Create placed trees
    n_placed = 20
    placed_trees = []
    for i in range(n_placed):
        tree = create_simple_tree()
        tree.set_position(
            Decimal(str(np.random.randn() * 10)),
            Decimal(str(np.random.randn() * 10)),
            Decimal(str(np.random.uniform(0, 360)))
        )
        placed_trees.append(tree)
    
    print(f"Setup: {n_placed} placed trees")
    print()
    
    # Test with different batch sizes
    batch_sizes = [10, 50, 100, 200]
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size} test positions")
        print("-" * 70)
        
        # Generate test positions and angles
        test_positions = np.random.randn(batch_size, 2).astype(np.float32) * 15
        test_angles = np.random.uniform(0, 360, batch_size).astype(np.float32)
        
        # Method 1: Original GPU batch (with individual polygon checks)
        try:
            from gpu_optimizations import check_collision_gpu_batch
            
            start = time.time()
            results1 = check_collision_gpu_batch(test_positions, test_angles, placed_trees, tree_template)
            time1 = time.time() - start
            
            print(f"  Original GPU batch:     {time1*1000:6.2f} ms  ({np.sum(results1)} collisions)")
        except Exception as e:
            print(f"  Original GPU batch:     ERROR - {e}")
            time1 = None
            results1 = None
        
        # Method 2: Optimized batch with caching
        try:
            from gpu_polygon_batch import optimized_collision_batch_with_caching
            
            start = time.time()
            results2 = optimized_collision_batch_with_caching(test_positions, test_angles, placed_trees, tree_template)
            time2 = time.time() - start
            
            print(f"  Optimized with caching: {time2*1000:6.2f} ms  ({np.sum(results2)} collisions)")
            
            if results1 is not None:
                speedup = time1 / time2 if time2 > 0 else 0
                print(f"  Speedup vs original:    {speedup:.2f}x")
        except Exception as e:
            print(f"  Optimized with caching: ERROR - {e}")
            time2 = None
            results2 = None
        
        # Method 3: CPU fallback (for comparison)
        try:
            from gpu_optimizations import check_collision_cpu_batch
            
            start = time.time()
            results3 = check_collision_cpu_batch(test_positions, test_angles, placed_trees, tree_template)
            time3 = time.time() - start
            
            print(f"  CPU fallback:           {time3*1000:6.2f} ms  ({np.sum(results3)} collisions)")
            
            if time2 is not None:
                speedup_vs_cpu = time3 / time2 if time2 > 0 else 0
                print(f"  GPU speedup vs CPU:     {speedup_vs_cpu:.2f}x")
        except Exception as e:
            print(f"  CPU fallback:           ERROR - {e}")
        
        # Verify results match
        if results1 is not None and results2 is not None and results3 is not None:
            match_12 = np.array_equal(results1, results2)
            match_23 = np.array_equal(results2, results3)
            print(f"  Results match:          {'✓' if match_12 and match_23 else '✗'}")
        
        print()
    
    # GPU memory info
    try:
        from gpu_polygon_batch import get_gpu_memory_info
        print(get_gpu_memory_info())
    except:
        pass
    
    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        benchmark_collision_detection()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
