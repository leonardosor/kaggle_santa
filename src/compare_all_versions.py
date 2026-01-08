"""Compare all algorithm versions: baseline, optimized, fast, and enhanced."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_scores(filepath):
    """Load scores from CSV file."""
    if not filepath.exists():
        return None
    return pd.read_csv(filepath)


def compare_all_versions():
    """Compare all available algorithm versions."""
    base_dir = Path(__file__).parent.parent / 'output'

    # Load all versions
    versions = {
        'Baseline': base_dir / 'scores_baseline.csv',
        'Optimized Fast': base_dir / 'scores_optimized_fast.csv',
        'Optimized Full': base_dir / 'scores_optimized.csv',
        'Enhanced': base_dir / 'scores_enhanced.csv',
    }

    scores_data = {}
    for name, path in versions.items():
        scores = load_scores(path)
        if scores is not None:
            scores_data[name] = scores

    if 'Baseline' not in scores_data:
        print("ERROR: Baseline scores not found. Run run_baseline.py first.")
        return

    if len(scores_data) == 1:
        print("ERROR: No optimized versions found. Run optimization scripts first.")
        return

    print("=" * 100)
    print("COMPREHENSIVE COMPARISON OF ALL ALGORITHMS")
    print("=" * 100)

    # Calculate total scores
    print("\n" + "=" * 100)
    print("TOTAL SCORES")
    print("=" * 100)
    print(f"{'Algorithm':<20} {'Total Score':>15} {'vs Baseline':>15} {'Improvement':>12}")
    print("-" * 100)

    baseline_total = scores_data['Baseline']['side_length'].sum()

    results = []
    for name in ['Baseline', 'Optimized Fast', 'Optimized Full', 'Enhanced']:
        if name not in scores_data:
            continue

        total = scores_data[name]['side_length'].sum()
        diff = baseline_total - total
        pct = (diff / baseline_total) * 100

        print(f"{name:<20} {total:>15.6f} {diff:>+15.6f} {pct:>+11.2f}%")
        results.append((name, total, diff, pct))

    # Find best version
    print("\n" + "=" * 100)
    best_name, best_score, best_diff, best_pct = min(results[1:], key=lambda x: x[1])
    print(f"BEST ALGORITHM: {best_name}")
    print(f"  Score: {best_score:.6f}")
    print(f"  Improvement: {best_diff:.6f} ({best_pct:.2f}%)")
    print("=" * 100)

    # Per-configuration comparison
    print("\n" + "=" * 100)
    print("IMPROVEMENT BY TREE COUNT RANGE")
    print("=" * 100)

    ranges = [
        (1, 10, "1-10 trees"),
        (11, 25, "11-25 trees"),
        (26, 50, "26-50 trees"),
        (51, 100, "51-100 trees"),
        (101, 150, "101-150 trees"),
        (151, 200, "151-200 trees")
    ]

    for start, end, label in ranges:
        print(f"\n{label}:")
        print(f"  {'Algorithm':<20} {'Avg Score':>12} {'Improvement':>12}")
        print("  " + "-" * 50)

        baseline_subset = scores_data['Baseline'][
            (scores_data['Baseline']['num_trees'] >= start) &
            (scores_data['Baseline']['num_trees'] <= end)
        ]['side_length'].mean()

        for name in ['Baseline', 'Optimized Fast', 'Optimized Full', 'Enhanced']:
            if name not in scores_data:
                continue

            subset = scores_data[name][
                (scores_data[name]['num_trees'] >= start) &
                (scores_data[name]['num_trees'] <= end)
            ]['side_length'].mean()

            improvement = ((baseline_subset - subset) / baseline_subset) * 100
            print(f"  {name:<20} {subset:>12.6f} {improvement:>+11.2f}%")

    # Configuration-by-configuration comparison
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGURATIONS WITH MOST IMPROVEMENT")
    print("=" * 100)

    # Build comparison dataframe
    comparison = pd.DataFrame({
        'num_trees': scores_data['Baseline']['num_trees']
    })

    for name, data in scores_data.items():
        comparison[name] = data['side_length']

    # Calculate improvements vs baseline
    for name in scores_data.keys():
        if name != 'Baseline':
            comparison[f'{name}_improvement'] = \
                ((comparison['Baseline'] - comparison[name]) / comparison['Baseline']) * 100

    # Find configurations with best improvements
    improvement_cols = [col for col in comparison.columns if col.endswith('_improvement')]

    if improvement_cols:
        # For each algorithm, show top improvements
        for imp_col in improvement_cols:
            algo_name = imp_col.replace('_improvement', '')
            print(f"\n{algo_name}:")
            top_configs = comparison.nlargest(10, imp_col)[['num_trees', 'Baseline', algo_name, imp_col]]
            top_configs.columns = ['Trees', 'Baseline', algo_name, 'Improvement %']
            print(top_configs.to_string(index=False))

    # Save full comparison
    comparison_path = base_dir / 'comparison_all.csv'
    comparison.to_csv(comparison_path, index=False)
    print(f"\n\nFull comparison saved to {comparison_path}")

    # Generate plots
    create_comparison_plots(scores_data, comparison, base_dir)


def create_comparison_plots(scores_data, comparison, output_dir):
    """Create comprehensive comparison plots."""
    print("\n" + "=" * 100)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 100)

    # Plot 1: All algorithms side length comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        'Baseline': 'red',
        'Optimized Fast': 'blue',
        'Optimized Full': 'green',
        'Enhanced': 'purple'
    }

    for name, data in scores_data.items():
        ax.plot(data['num_trees'], data['side_length'],
                label=name, alpha=0.7, linewidth=2, color=colors.get(name, 'gray'))

    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Bounding Square Side Length', fontsize=12)
    ax.set_title('Algorithm Comparison: Side Length vs Tree Count', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot1_path = output_dir / 'comparison_all_algorithms.png'
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot1_path}")
    plt.close()

    # Plot 2: Improvement percentage over baseline
    fig, ax = plt.subplots(figsize=(14, 7))

    improvement_cols = [col for col in comparison.columns if col.endswith('_improvement')]
    colors_list = ['blue', 'green', 'purple', 'orange']

    for i, col in enumerate(improvement_cols):
        algo_name = col.replace('_improvement', '')
        ax.plot(comparison['num_trees'], comparison[col],
                label=algo_name, alpha=0.7, linewidth=2, color=colors_list[i % len(colors_list)])

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Percentage Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot2_path = output_dir / 'comparison_improvements.png'
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot2_path}")
    plt.close()

    # Plot 3: Box plot of improvements by range
    fig, ax = plt.subplots(figsize=(12, 7))

    ranges = [
        (1, 10), (11, 25), (26, 50), (51, 100), (101, 150), (151, 200)
    ]
    range_labels = ['1-10', '11-25', '26-50', '51-100', '101-150', '151-200']

    box_data = []
    for start, end in ranges:
        range_improvements = []
        for col in improvement_cols:
            subset = comparison[
                (comparison['num_trees'] >= start) &
                (comparison['num_trees'] <= end)
            ][col]
            range_improvements.extend(subset.tolist())
        box_data.append(range_improvements)

    bp = ax.boxplot(box_data, labels=range_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Tree Count Range', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Distribution of Improvements by Tree Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plot3_path = output_dir / 'comparison_distribution.png'
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot3_path}")
    plt.close()

    # Plot 4: Cumulative score comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, data in scores_data.items():
        cumulative = data['side_length'].cumsum()
        ax.plot(data['num_trees'], cumulative,
                label=name, alpha=0.7, linewidth=2, color=colors.get(name, 'gray'))

    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Cumulative Score', fontsize=12)
    ax.set_title('Cumulative Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot4_path = output_dir / 'comparison_cumulative.png'
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot4_path}")
    plt.close()


if __name__ == "__main__":
    compare_all_versions()
