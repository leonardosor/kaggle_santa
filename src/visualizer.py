"""
Visualization tools for Santa's tree packing

Visualize packing solutions from submission CSV files
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union


# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')


def create_tree_polygon(center_x, center_y, angle):
    """Create a tree polygon with given position and rotation"""
    if isinstance(center_x, str) and center_x.startswith('s'):
        center_x = center_x.lstrip('s')
    if isinstance(center_y, str) and center_y.startswith('s'):
        center_y = center_y.lstrip('s')
    if isinstance(angle, str) and angle.startswith('s'):
        angle = angle.lstrip('s')

    center_x = Decimal(center_x)
    center_y = Decimal(center_y)
    angle = Decimal(angle)

    # Tree dimensions
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

    rotated = affinity.rotate(initial_polygon, float(angle), origin=(0, 0))
    polygon = affinity.translate(rotated,
                                xoff=float(center_x * scale_factor),
                                yoff=float(center_y * scale_factor))

    return polygon


def visualize_configuration(csv_path: str, n_trees: int, save_path: str = None):
    """
    Visualize a specific n-tree configuration

    Args:
        csv_path: Path to submission CSV
        n_trees: Number of trees to visualize (1-200)
        save_path: Optional path to save figure
    """
    # Load submission
    df = pd.read_csv(csv_path)

    # Get rows for this configuration
    config_id_prefix = f'{n_trees:03d}_'
    config_df = df[df['id'].str.startswith(config_id_prefix)]

    if len(config_df) != n_trees:
        print(f"Error: Expected {n_trees} trees, found {len(config_df)}")
        return

    # Create tree polygons
    trees = []
    for _, row in config_df.iterrows():
        polygon = create_tree_polygon(row['x'], row['y'], row['deg'])
        trees.append(polygon)

    # Calculate bounding box
    bounds = unary_union(trees).bounds
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.viridis([i / n_trees for i in range(n_trees)])

    for i, tree in enumerate(trees):
        x_scaled, y_scaled = tree.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])
        ax.fill(x, y, alpha=0.5, color=colors[i])

    # Draw bounding square
    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2

    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--',
    )
    ax.add_patch(bounding_square)

    # Format plot
    padding = 0.5
    ax.set_xlim(float(square_x - Decimal(str(padding))),
                float(square_x + side_length + Decimal(str(padding))))
    ax.set_ylim(float(square_y - Decimal(str(padding))),
                float(square_y + side_length + Decimal(str(padding))))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    score = (side_length ** 2) / Decimal(n_trees)
    plt.title(f'{n_trees} Trees | Side: {side_length:.6f} | Score: {float(score):.6f}',
              fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_multiple(csv_path: str, n_list: list, save_path: str = None):
    """
    Visualize multiple configurations in a grid

    Args:
        csv_path: Path to submission CSV
        n_list: List of tree counts to visualize
        save_path: Optional path to save figure
    """
    df = pd.read_csv(csv_path)

    num_configs = len(n_list)
    cols = min(3, num_configs)
    rows = (num_configs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

    if num_configs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, n_trees in enumerate(n_list):
        ax = axes[idx] if num_configs > 1 else axes[0]

        # Get configuration
        config_id_prefix = f'{n_trees:03d}_'
        config_df = df[df['id'].str.startswith(config_id_prefix)]

        # Create trees
        trees = []
        for _, row in config_df.iterrows():
            polygon = create_tree_polygon(row['x'], row['y'], row['deg'])
            trees.append(polygon)

        # Calculate bounds
        bounds = unary_union(trees).bounds
        minx = Decimal(bounds[0]) / scale_factor
        miny = Decimal(bounds[1]) / scale_factor
        maxx = Decimal(bounds[2]) / scale_factor
        maxy = Decimal(bounds[3]) / scale_factor

        width = maxx - minx
        height = maxy - miny
        side_length = max(width, height)

        # Plot trees
        colors = plt.cm.viridis([i / n_trees for i in range(n_trees)])
        for i, tree in enumerate(trees):
            x_scaled, y_scaled = tree.exterior.xy
            x = [Decimal(val) / scale_factor for val in x_scaled]
            y = [Decimal(val) / scale_factor for val in y_scaled]
            ax.plot(x, y, color=colors[i], linewidth=0.5)
            ax.fill(x, y, alpha=0.5, color=colors[i])

        # Draw bounding square
        square_x = minx if width >= height else minx - (side_length - width) / 2
        square_y = miny if height >= width else miny - (side_length - height) / 2

        ax.add_patch(Rectangle(
            (float(square_x), float(square_y)),
            float(side_length),
            float(side_length),
            fill=False,
            edgecolor='red',
            linewidth=2,
            linestyle='--',
        ))

        # Format
        padding = 0.3
        ax.set_xlim(float(square_x - Decimal(str(padding))),
                    float(square_x + side_length + Decimal(str(padding))))
        ax.set_ylim(float(square_y - Decimal(str(padding))),
                    float(square_y + side_length + Decimal(str(padding))))
        ax.set_aspect('equal')
        ax.axis('off')

        score = (side_length ** 2) / Decimal(n_trees)
        ax.set_title(f'n={n_trees} | {float(score):.3f}', fontsize=11)

    # Hide unused subplots
    for idx in range(num_configs, len(axes) if isinstance(axes, list) else 1):
        if isinstance(axes, list):
            axes[idx].axis('off')

    plt.suptitle('Santa Tree Packing Solutions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Santa 2025 packing solutions"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to submission CSV file"
    )
    parser.add_argument(
        "--n-trees",
        type=int,
        default=None,
        help="Number of trees to visualize (single configuration)"
    )
    parser.add_argument(
        "--multiple",
        type=int,
        nargs='+',
        default=None,
        help="Multiple configurations to visualize in grid"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save figure"
    )

    args = parser.parse_args()

    if args.n_trees:
        visualize_configuration(args.csv_path, args.n_trees, args.save)
    elif args.multiple:
        visualize_multiple(args.csv_path, args.multiple, args.save)
    else:
        # Default: show a few interesting configurations
        print("Visualizing configurations: 5, 10, 25, 50, 100, 200")
        visualize_multiple(args.csv_path, [5, 10, 25, 50, 100, 200], args.save)


if __name__ == "__main__":
    main()
