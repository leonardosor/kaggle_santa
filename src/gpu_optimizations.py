"""GPU-accelerated optimization strategies using CuPy and multi-GPU support."""
import numpy as np

try:
    import cupy as cp
    import cupyx
    CUDA_AVAILABLE = True
    print("CUDA/CuPy available for GPU acceleration")
except ImportError:
    CUDA_AVAILABLE = False
    cp = np
    print("WARNING: CuPy not available. Falling back to NumPy (CPU only)")
    print("Install CuPy with: pip install cupy-cuda11x  # or cupy-cuda12x")

from decimal import Decimal
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity
import concurrent.futures
from run_optimized import ChristmasTree, scale_factor
import gpu_polygon_ops
import gpu_polygon_batch
import gpu_adaptive


# ============================================================================
# GPU-Accelerated Collision Detection
# ============================================================================

def trees_to_numpy_polygons(trees):
    """
    Convert tree polygons to numpy arrays for GPU processing.
    Returns arrays of polygon vertices.
    """
    polygons_data = []

    for tree in trees:
        coords = list(tree.polygon.exterior.coords)
        # Convert to numpy array
        coords_array = np.array(coords, dtype=np.float32)
        polygons_data.append(coords_array)

    return polygons_data


def check_collision_gpu_batch(test_positions, test_angles, placed_trees, tree_template):
    """
    Batch collision detection with adaptive GPU/CPU selection.
    Automatically chooses GPU or CPU based on workload size.

    Args:
        test_positions: Nx2 array of (x, y) positions to test
        test_angles: N array of angles to test
        placed_trees: List of already placed trees
        tree_template: Template tree for creating test polygons

    Returns:
        Boolean array indicating which positions have collisions
    """
    # Use adaptive selection for optimal performance
    return gpu_adaptive.adaptive_collision_check(
        test_positions, test_angles, placed_trees, tree_template
    )


def check_collision_cpu_batch(test_positions, test_angles, placed_trees, tree_template):
    """CPU fallback for batch collision detection using GPU polygon ops (CPU fallback)."""
    collisions = np.zeros(len(test_positions), dtype=bool)

    # Convert placed trees to vertex arrays once
    placed_polygon_verts = [gpu_polygon_ops.polygon_to_array(t.polygon) for t in placed_trees]

    for i in range(len(test_positions)):
        test_tree = tree_template.copy()
        test_tree.set_position(
            Decimal(str(test_positions[i, 0])),
            Decimal(str(test_positions[i, 1])),
            Decimal(str(test_angles[i]))
        )
        
        test_poly_verts = gpu_polygon_ops.polygon_to_array(test_tree.polygon)

        for placed_verts in placed_polygon_verts:
            intersects, touches = gpu_polygon_ops.polygons_intersect_cpu(
                test_poly_verts, placed_verts
            )
            if intersects and not touches:
                collisions[i] = True
                break

    return collisions


# ============================================================================
# GPU-Accelerated Bounding Box Calculation
# ============================================================================

def compute_bounding_box_gpu(trees):
    """
    Compute bounding box for multiple tree configurations on GPU.
    Much faster than CPU for large numbers of trees.
    Uses the currently active GPU device and GPU polygon operations.
    """
    if not CUDA_AVAILABLE or len(trees) < 10:
        # For small counts, CPU is fine
        from run_optimized import get_bounding_square
        return get_bounding_square([t.polygon for t in trees])

    # Convert trees to vertex arrays
    polygon_verts = [gpu_polygon_ops.polygon_to_array(t.polygon) for t in trees]
    
    # Compute bounds using GPU
    minx, miny, maxx, maxy = gpu_polygon_ops.compute_polygon_bounds_gpu(polygon_verts)

    # Scale back
    minx_scaled = Decimal(minx) / scale_factor
    miny_scaled = Decimal(miny) / scale_factor
    maxx_scaled = Decimal(maxx) / scale_factor
    maxy_scaled = Decimal(maxy) / scale_factor

    width = maxx_scaled - minx_scaled
    height = maxy_scaled - miny_scaled

    return max(width, height)


# ============================================================================
# Multi-GPU Parallel Tree Placement
# ============================================================================

def place_tree_parallel_gpu(tree_to_place, placed_trees, num_attempts=250, gpu_device=0):
    """
    Place a tree using GPU-accelerated search on the specified GPU.
    Increased num_attempts to 250 for optimal RTX 5070 GPU utilization.

    Args:
        tree_to_place: Tree to place
        placed_trees: Already placed trees
        num_attempts: Number of random attempts (default 250 for RTX 5070)
        gpu_device: Which GPU device to use

    Returns:
        Best placement found
    """
    if not placed_trees:
        tree_to_place.set_position(Decimal('0'), Decimal('0'))
        return tree_to_place

    if not CUDA_AVAILABLE:
        # Fall back to CPU
        from run_optimized import place_tree_greedy
        return place_tree_greedy(tree_to_place, placed_trees, num_attempts=num_attempts)

    # Ensure we're using the correct GPU
    cp.cuda.Device(gpu_device).use()

    # Generate candidate positions and angles
    np.random.seed()
    angles_rad = np.random.uniform(0, 2 * np.pi, num_attempts)

    # Weight by sin(2*angle) for corner bias
    weights = np.abs(np.sin(2 * angles_rad))
    weights = weights / np.sum(weights)
    selected_angles = np.random.choice(angles_rad, size=num_attempts, p=weights)

    candidate_angles = np.random.uniform(0, 360, num_attempts).astype(np.float32)

    # Generate candidate positions (on circles moving inward)
    radii = np.linspace(25, 0.1, num_attempts).astype(np.float32)
    vx = np.cos(selected_angles).astype(np.float32)
    vy = np.sin(selected_angles).astype(np.float32)

    candidate_positions = np.stack([
        radii * vx,
        radii * vy
    ], axis=1)

    # Batch collision detection
    collisions = check_collision_gpu_batch(
        candidate_positions,
        candidate_angles,
        placed_trees,
        tree_to_place
    )

    # Find best non-colliding position (closest to center)
    valid_mask = ~collisions
    if not np.any(valid_mask):
        # All collide, find least bad
        valid_idx = 0
    else:
        valid_radii = radii[valid_mask]
        best_valid_idx = np.argmin(valid_radii)
        valid_indices = np.where(valid_mask)[0]
        valid_idx = valid_indices[best_valid_idx]

    best_pos = candidate_positions[valid_idx]
    best_angle = candidate_angles[valid_idx]

    tree_to_place.set_position(
        Decimal(str(best_pos[0])),
        Decimal(str(best_pos[1])),
        Decimal(str(best_angle))
    )

    return tree_to_place


# ============================================================================
# Multi-GPU Parallel Rotation Optimization
# ============================================================================

def optimize_rotation_parallel_gpu(tree_idx, placed_trees, angle_steps=72, gpu_device=0):
    """
    Optimize tree rotation using GPU evaluation on the specified GPU.

    Args:
        tree_idx: Index of tree to optimize
        placed_trees: All placed trees
        angle_steps: Number of angles to test (72 = 5 degree steps)
        gpu_device: Which GPU device to use

    Returns:
        True if improvement found
    """
    tree = placed_trees[tree_idx]
    other_trees = [t for i, t in enumerate(placed_trees) if i != tree_idx]

    if not CUDA_AVAILABLE or not other_trees:
        # Fall back to CPU
        from run_optimized import optimize_tree_rotation
        return optimize_tree_rotation(tree_idx, placed_trees, angle_step=5)

    # Ensure we're using the correct GPU
    cp.cuda.Device(gpu_device).use()

    # Generate test angles
    test_angles = np.linspace(0, 360, angle_steps, endpoint=False).astype(np.float32)

    # Fixed position, varying angles
    test_positions = np.tile(
        [[float(tree.center_x), float(tree.center_y)]],
        (angle_steps, 1)
    ).astype(np.float32)

    # Batch collision check
    collisions = check_collision_gpu_batch(
        test_positions,
        test_angles,
        other_trees,
        tree
    )

    # For each non-colliding angle, compute bounding box
    # This is still expensive, but we can parallelize it

    current_polygons = [t.polygon for t in placed_trees]
    from run_optimized import get_bounding_square
    current_side = get_bounding_square(current_polygons)

    best_angle = tree.angle
    best_side = current_side
    improved = False

    valid_angles = test_angles[~collisions]

    if len(valid_angles) > 0:
        # Test each valid angle (still needs CPU for bounding box)
        for angle in valid_angles:
            temp_tree = tree.copy()
            temp_tree.set_position(tree.center_x, tree.center_y, Decimal(str(angle)))

            test_polygons = list(current_polygons)
            test_polygons[tree_idx] = temp_tree.polygon
            test_side = get_bounding_square(test_polygons)

            if test_side < best_side - Decimal('1e-10'):
                best_side = test_side
                best_angle = Decimal(str(angle))
                improved = True

    if improved:
        tree.set_position(tree.center_x, tree.center_y, best_angle)

    return improved


# ============================================================================
# Multi-GPU Parallel Configuration Evaluation
# ============================================================================

def evaluate_configurations_single_gpu(configurations, gpu_device=0):
    """
    Evaluate multiple tree configurations on a single GPU.

    Args:
        configurations: List of (trees, label) tuples
        gpu_device: Which GPU device to use

    Returns:
        List of (score, label) tuples
    """
    if not CUDA_AVAILABLE:
        # CPU fallback
        from run_optimized import get_bounding_square
        results = []
        for trees, label in configurations:
            score = get_bounding_square([t.polygon for t in trees])
            results.append((score, label))
        return results

    # Ensure we're using the correct GPU
    cp.cuda.Device(gpu_device).use()
    results = []

    for trees, label in configurations:
        score = compute_bounding_box_gpu(trees)
        results.append((score, label))

    return results


def evaluate_configurations_multi_gpu(configurations, num_gpus=2):
    """
    Evaluate multiple tree configurations in parallel across GPUs.
    NOTE: This function is deprecated. Use evaluate_configurations_single_gpu instead.

    Args:
        configurations: List of (trees, label) tuples
        num_gpus: Number of GPUs to use

    Returns:
        List of (score, label) tuples
    """
    if not CUDA_AVAILABLE:
        # CPU fallback
        from run_optimized import get_bounding_square
        results = []
        for trees, label in configurations:
            score = get_bounding_square([t.polygon for t in trees])
            results.append((score, label))
        return results

    # Distribute configurations across GPUs
    results = []

    def evaluate_on_gpu(gpu_id, configs):
        cp.cuda.Device(gpu_id % num_gpus).use()
        local_results = []

        for trees, label in configs:
            score = compute_bounding_box_gpu(trees)
            local_results.append((score, label))

        return local_results

    # Split configurations across GPUs
    configs_per_gpu = len(configurations) // num_gpus
    gpu_configs = [
        configurations[i*configs_per_gpu:(i+1)*configs_per_gpu]
        for i in range(num_gpus)
    ]

    # Add remaining to last GPU
    if len(configurations) % num_gpus != 0:
        gpu_configs[-1].extend(configurations[num_gpus*configs_per_gpu:])

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(evaluate_on_gpu, gpu_id, configs)
            for gpu_id, configs in enumerate(gpu_configs)
        ]

        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results


# ============================================================================
# GPU-Accelerated Simulated Annealing
# ============================================================================

def simulated_annealing_gpu(placed_trees, initial_temp=1.0, cooling_rate=0.95,
                           max_iterations=200, gpu_device=0):
    """
    GPU-accelerated simulated annealing on the specified GPU.
    """
    # Ensure we're using the correct GPU
    if CUDA_AVAILABLE:
        cp.cuda.Device(gpu_device).use()

    current_trees = [t.copy() for t in placed_trees]
    current_side = compute_bounding_box_gpu(current_trees)

    best_trees = [t.copy() for t in current_trees]
    best_side = current_side

    temp = initial_temp
    num_trees = len(placed_trees)

    print(f"  Starting GPU-accelerated SA on GPU {gpu_device}...", flush=True)

    for iteration in range(max_iterations):
        # Generate multiple candidate perturbations in parallel
        candidates_per_iter = min(10, num_trees)

        candidate_configs = []

        for _ in range(candidates_per_iter):
            # Random perturbation
            tree_idx = np.random.randint(0, num_trees)
            test_trees = [t.copy() for t in current_trees]
            tree = test_trees[tree_idx]

            # Perturb position or angle
            if np.random.random() < 0.5:
                delta_x = Decimal(str(np.random.uniform(-0.5, 0.5))) * Decimal(str(temp))
                delta_y = Decimal(str(np.random.uniform(-0.5, 0.5))) * Decimal(str(temp))
                tree.set_position(tree.center_x + delta_x, tree.center_y + delta_y)
            else:
                delta_angle = Decimal(str(np.random.uniform(-30, 30))) * Decimal(str(temp))
                tree.set_position(tree.center_x, tree.center_y, tree.angle + delta_angle)

            candidate_configs.append((test_trees, tree_idx))

        # Evaluate all candidates on the GPU
        evaluations = evaluate_configurations_single_gpu(candidate_configs, gpu_device)

        # Select best acceptable candidate
        for (candidate_trees, tree_idx), (new_side, _) in zip(candidate_configs, evaluations):
            # Check collision
            tree = candidate_trees[tree_idx]
            other_trees = [t for i, t in enumerate(candidate_trees) if i != tree_idx]
            other_polygons = [t.polygon for t in other_trees]

            from run_optimized import check_collision
            if check_collision(tree.polygon, other_polygons):
                continue

            # Metropolis criterion
            delta_side = float(new_side - current_side)

            if delta_side < 0 or np.random.random() < np.exp(-delta_side / temp):
                current_trees = [t.copy() for t in candidate_trees]
                current_side = new_side

                if new_side < best_side:
                    best_trees = [t.copy() for t in current_trees]
                    best_side = new_side

                break  # Accept first good candidate

        # Cool down
        if (iteration + 1) % 20 == 0:
            temp *= cooling_rate
            print(f"    GPU-SA iteration {iteration + 1}: temp = {temp:.4f}, best = {best_side:.6f}", flush=True)

    return best_trees


# ============================================================================
# Main GPU-Accelerated Initialization
# ============================================================================

def initialize_trees_gpu(num_trees, existing_trees=None, num_gpus=1, gpu_device=0, use_optimization=True):
    """
    Main initialization function with GPU acceleration.

    Args:
        num_trees: Number of trees to place
        existing_trees: Previously placed trees (for incremental building)
        num_gpus: Number of GPUs to use (should be 1 to use primary GPU)
        gpu_device: Which GPU device to use (the one with most memory)
        use_optimization: Whether to apply optimization passes

    Returns:
        (placed_trees, bounding_box_side_length)
    """
    if num_trees == 0:
        return [], Decimal('0')

    # Set the GPU device to use
    if CUDA_AVAILABLE and num_gpus > 0:
        cp.cuda.Device(gpu_device).use()
        print(f"  Using GPU {gpu_device} for acceleration", flush=True)
    else:
        print(f"  Using CPU (no GPU acceleration)", flush=True)

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = [t.copy() for t in existing_trees]

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        if not placed_trees:
            first_tree = ChristmasTree(angle=str(np.random.uniform(0, 360)))
            placed_trees.append(first_tree)
            num_to_add -= 1

        # Place remaining trees with GPU acceleration
        for i in range(num_to_add):
            tree = ChristmasTree(angle=str(np.random.uniform(0, 360)))
            place_tree_parallel_gpu(tree, placed_trees, num_attempts=250, gpu_device=gpu_device)
            placed_trees.append(tree)

    # GPU-accelerated optimization
    if use_optimization and num_trees >= 10:
        print(f"  Running GPU-accelerated optimization...", flush=True)

        # Parallel rotation optimization
        for idx in range(len(placed_trees)):
            optimize_rotation_parallel_gpu(idx, placed_trees, angle_steps=72, gpu_device=gpu_device)

        # GPU-accelerated simulated annealing every 10 trees
        if num_trees % 10 == 0:
            placed_trees = simulated_annealing_gpu(
                placed_trees,
                initial_temp=0.5,
                cooling_rate=0.95,
                max_iterations=50,
                gpu_device=gpu_device
            )

    side_length = compute_bounding_box_gpu(placed_trees)

    return placed_trees, side_length
