"""Adaptive GPU usage based on workload size."""
import numpy as np
import sys


# Thresholds for adaptive GPU usage (tuned for RTX 5070)
MIN_BATCH_SIZE_FOR_GPU = 20  # Minimum batch size to use GPU
MIN_PLACED_TREES_FOR_GPU = 5  # Minimum number of placed trees to use GPU (lowered for faster GPU activation)

# Enable memory pooling and CUDA kernels by default
USE_MEMORY_POOL = True
USE_CUDA_KERNELS = True  # Use custom CUDA kernels for maximum performance

# Track if we've logged the GPU configuration
_gpu_config_logged = False


def should_use_gpu(batch_size, num_placed_trees):
    """
    Determine if GPU should be used based on workload size.
    
    GPU has overhead from:
    - CPU-GPU memory transfers
    - Kernel launch latency
    - Data conversion
    
    For small workloads, CPU is faster.
    
    Args:
        batch_size: Number of test positions to check
        num_placed_trees: Number of already placed trees
        
    Returns:
        bool: True if GPU should be used
    """
    try:
        import cupy as cp
        gpu_available = True
    except ImportError:
        return False
    
    # Need sufficient work to amortize GPU overhead
    if batch_size < MIN_BATCH_SIZE_FOR_GPU:
        return False
    
    if num_placed_trees < MIN_PLACED_TREES_FOR_GPU:
        return False
    
    # Total work = batch_size * num_placed_trees
    total_checks = batch_size * num_placed_trees
    
    # Use GPU if total work exceeds threshold
    # 200 checks = ~10ms on GPU vs ~50ms on CPU
    return total_checks >= 200


def adaptive_collision_check(test_positions, test_angles, placed_trees, tree_template):
    """
    Automatically choose GPU or CPU based on workload size.
    Uses CUDA kernels and memory pooling when GPU is selected for maximum performance.
    
    Args:
        test_positions: Nx2 array of positions to test
        test_angles: N array of angles to test
        placed_trees: List of placed trees
        tree_template: Template tree
        
    Returns:
        Boolean array indicating collisions
    """
    global _gpu_config_logged
    batch_size = len(test_positions)
    num_placed = len(placed_trees)
    
    use_gpu = should_use_gpu(batch_size, num_placed)
    
    # Log GPU configuration on first use
    if not _gpu_config_logged:
        _gpu_config_logged = True
        if use_gpu:
            print("\n" + "="*60)
            print("[GPU] Acceleration Configuration")
            print("="*60)
            if USE_CUDA_KERNELS:
                print("   Mode: CUDA Kernels (Maximum Performance)")
            elif USE_MEMORY_POOL:
                print("   Mode: Memory Pool (High Performance)")
            else:
                print("   Mode: Standard GPU Batch")
            print(f"   Threshold: {MIN_BATCH_SIZE_FOR_GPU}+ positions, {MIN_PLACED_TREES_FOR_GPU}+ trees")
            print(f"   Current batch: {batch_size} positions, {num_placed} trees")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("[CPU] Using CPU mode (workload below GPU threshold)")
            print("="*60 + "\n")
        sys.stdout.flush()
    
    if use_gpu and USE_CUDA_KERNELS:
        # Use custom CUDA kernels for maximum performance
        try:
            from cuda_kernels import cuda_batch_collision_check
            return cuda_batch_collision_check(
                test_positions, test_angles, placed_trees, tree_template
            )
        except Exception as e:
            # Fall back to memory pool if CUDA kernels fail
            print(f"[WARN] CUDA kernels unavailable: {e}")
            print("   Falling back to memory pool mode...\n")
            sys.stdout.flush()
            pass
    
    if use_gpu and USE_MEMORY_POOL:
        # Use memory pool version for best performance
        from gpu_memory_pool import optimized_batch_collision_with_memory_pool
        return optimized_batch_collision_with_memory_pool(
            test_positions, test_angles, placed_trees, tree_template
        )
    elif use_gpu:
        # Use standard GPU version without memory pool
        from gpu_polygon_batch import optimized_collision_batch_with_caching
        return optimized_collision_batch_with_caching(
            test_positions, test_angles, placed_trees, tree_template
        )
    else:
        # Use CPU version
        from gpu_polygon_batch import optimized_collision_batch_cpu
        return optimized_collision_batch_cpu(
            test_positions, test_angles, placed_trees, tree_template
        )


def get_performance_stats():
    """Get statistics on GPU vs CPU usage."""
    stats = {
        'gpu_calls': 0,
        'cpu_calls': 0,
        'gpu_time': 0.0,
        'cpu_time': 0.0
    }
    return stats


# Global stats tracking
_performance_stats = get_performance_stats()


def log_performance(method, duration):
    """Log performance for analysis."""
    global _performance_stats
    if method == 'gpu':
        _performance_stats['gpu_calls'] += 1
        _performance_stats['gpu_time'] += duration
    else:
        _performance_stats['cpu_calls'] += 1
        _performance_stats['cpu_time'] += duration


def print_performance_summary():
    """Print summary of GPU vs CPU usage."""
    global _performance_stats
    
    total_calls = _performance_stats['gpu_calls'] + _performance_stats['cpu_calls']
    total_time = _performance_stats['gpu_time'] + _performance_stats['cpu_time']
    
    if total_calls == 0:
        print("No performance data collected")
        return
    
    print("\nPerformance Summary:")
    print("=" * 60)
    print(f"GPU calls: {_performance_stats['gpu_calls']} "
          f"({_performance_stats['gpu_calls']/total_calls*100:.1f}%)")
    print(f"CPU calls: {_performance_stats['cpu_calls']} "
          f"({_performance_stats['cpu_calls']/total_calls*100:.1f}%)")
    
    if _performance_stats['gpu_calls'] > 0:
        avg_gpu = _performance_stats['gpu_time'] / _performance_stats['gpu_calls']
        print(f"Avg GPU time: {avg_gpu*1000:.2f} ms")
    
    if _performance_stats['cpu_calls'] > 0:
        avg_cpu = _performance_stats['cpu_time'] / _performance_stats['cpu_calls']
        print(f"Avg CPU time: {avg_cpu*1000:.2f} ms")
    
    print(f"Total time: {total_time:.2f} s")
    print("=" * 60)
