"""Benchmark GPU memory pool vs standard implementation."""
import numpy as np
import time
from decimal import Decimal


def benchmark_memory_pool():
    """Compare memory pool performance vs standard GPU implementation."""
    print("=" * 70)
    print("GPU Memory Pool Performance Benchmark")
    print("=" * 70)
    print()
    
    # Create test data
    from run_optimized import ChristmasTree
    tree_template = ChristmasTree()
    
    # Create placed trees
    n_placed = 30
    placed_trees = []
    for i in range(n_placed):
        tree = ChristmasTree()
        tree.set_position(
            Decimal(str(np.random.randn() * 10)),
            Decimal(str(np.random.randn() * 10)),
            Decimal(str(np.random.uniform(0, 360)))
        )
        placed_trees.append(tree)
    
    print(f"Setup: {n_placed} placed trees")
    print()
    
    # Test with different batch sizes
    batch_sizes = [50, 100, 200]
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size} test positions")
        print("-" * 70)
        
        # Generate test positions
        test_positions = np.random.randn(batch_size, 2).astype(np.float32) * 15
        test_angles = np.random.uniform(0, 360, batch_size).astype(np.float32)
        
        # Method 1: Standard GPU batch (no memory pool)
        try:
            from gpu_polygon_batch import optimized_collision_batch_with_caching
            
            start = time.time()
            results1 = optimized_collision_batch_with_caching(
                test_positions, test_angles, placed_trees, tree_template
            )
            time1 = time.time() - start
            
            print(f"  Standard GPU:      {time1*1000:6.2f} ms  ({np.sum(results1)} collisions)")
        except Exception as e:
            print(f"  Standard GPU:      ERROR - {e}")
            time1 = None
            results1 = None
        
        # Method 2: GPU with memory pool
        try:
            from gpu_memory_pool import optimized_batch_collision_with_memory_pool
            
            # First run (cold cache)
            start = time.time()
            results2 = optimized_batch_collision_with_memory_pool(
                test_positions, test_angles, placed_trees, tree_template
            )
            time2_cold = time.time() - start
            
            print(f"  GPU + Memory Pool: {time2_cold*1000:6.2f} ms  ({np.sum(results2)} collisions) [cold cache]")
            
            # Second run (warm cache)
            start = time.time()
            results2_warm = optimized_batch_collision_with_memory_pool(
                test_positions, test_angles, placed_trees, tree_template
            )
            time2_warm = time.time() - start
            
            print(f"  GPU + Memory Pool: {time2_warm*1000:6.2f} ms  ({np.sum(results2_warm)} collisions) [warm cache]")
            
            if time1 is not None:
                speedup_cold = time1 / time2_cold if time2_cold > 0 else 0
                speedup_warm = time1 / time2_warm if time2_warm > 0 else 0
                print(f"  Speedup (cold):    {speedup_cold:.2f}x")
                print(f"  Speedup (warm):    {speedup_warm:.2f}x")
        except Exception as e:
            print(f"  GPU + Memory Pool: ERROR - {e}")
            import traceback
            traceback.print_exc()
        
        # Method 3: CPU fallback
        try:
            from gpu_polygon_batch import optimized_collision_batch_cpu
            
            start = time.time()
            results3 = optimized_collision_batch_cpu(
                test_positions, test_angles, placed_trees, tree_template
            )
            time3 = time.time() - start
            
            print(f"  CPU:               {time3*1000:6.2f} ms  ({np.sum(results3)} collisions)")
        except Exception as e:
            print(f"  CPU:               ERROR - {e}")
        
        print()
    
    # Print memory stats
    try:
        from gpu_memory_pool import print_memory_stats
        print_memory_stats()
    except:
        pass
    
    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        benchmark_memory_pool()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
