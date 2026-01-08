"""Compare baseline and optimized results."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_scores(filepath):
    """Load scores from CSV file."""
    if not filepath.exists():
        return None
    return pd.read_csv(filepath)


def compare_results():
    """Compare baseline and optimized results."""
    base_dir = Path(__file__).parent.parent / 'output'

    baseline_path = base_dir / 'scores_baseline.csv'
    optimized_path = base_dir / 'scores_optimized.csv'

    baseline_scores = load_scores(baseline_path)
    optimized_scores = load_scores(optimized_path)

    if baseline_scores is None:
        print("ERROR: Baseline scores not found. Run run_baseline.py first.")
        return

    if optimized_scores is None:
        print("ERROR: Optimized scores not found. Run run_optimized.py first.")
        return

    # Calculate statistics
    baseline_total = baseline_scores['side_length'].sum()
    optimized_total = optimized_scores['side_length'].sum()
    improvement = baseline_total - optimized_total
    improvement_pct = (improvement / baseline_total) * 100

    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nBaseline Total Score:  {baseline_total:>15.6f}")
    print(f"Optimized Total Score: {optimized_total:>15.6f}")
    print(f"Improvement:           {improvement:>15.6f} ({improvement_pct:>6.2f}%)")

    # Per-configuration statistics
    comparison_df = pd.DataFrame({
        'num_trees': baseline_scores['num_trees'],
        'baseline': baseline_scores['side_length'],
        'optimized': optimized_scores['side_length'],
    })
    comparison_df['improvement'] = comparison_df['baseline'] - comparison_df['optimized']
    comparison_df['improvement_pct'] = (comparison_df['improvement'] / comparison_df['baseline']) * 100

    print("\n" + "=" * 80)
    print("TOP 10 IMPROVEMENTS (by percentage)")
    print("=" * 80)
    top_improvements = comparison_df.nlargest(10, 'improvement_pct')
    print(top_improvements.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 10 IMPROVEMENTS (by absolute value)")
    print("=" * 80)
    top_absolute = comparison_df.nlargest(10, 'improvement')
    print(top_absolute.to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICS BY TREE COUNT RANGES")
    print("=" * 80)

    ranges = [
        (1, 10, "1-10 trees"),
        (11, 50, "11-50 trees"),
        (51, 100, "51-100 trees"),
        (101, 150, "101-150 trees"),
        (151, 200, "151-200 trees")
    ]

    for start, end, label in ranges:
        subset = comparison_df[(comparison_df['num_trees'] >= start) &
                              (comparison_df['num_trees'] <= end)]
        avg_improvement = subset['improvement'].mean()
        avg_improvement_pct = subset['improvement_pct'].mean()
        print(f"{label:20s}: Avg improvement = {avg_improvement:8.6f} ({avg_improvement_pct:6.2f}%)")

    # Save comparison
    comparison_path = base_dir / 'comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nFull comparison saved to {comparison_path}")

    # Create visualizations
    create_plots(comparison_df, base_dir)


def create_plots(comparison_df, output_dir):
    """Create comparison plots."""
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Plot 1: Side length comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['num_trees'], comparison_df['baseline'],
            label='Baseline', alpha=0.7, linewidth=2)
    ax.plot(comparison_df['num_trees'], comparison_df['optimized'],
            label='Optimized', alpha=0.7, linewidth=2)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Bounding Square Side Length')
    ax.set_title('Baseline vs Optimized Algorithm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot1_path = output_dir / 'comparison_side_lengths.png'
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=150)
    print(f"Plot saved: {plot1_path}")
    plt.close()

    # Plot 2: Improvement percentage
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['num_trees'], comparison_df['improvement_pct'],
            color='green', alpha=0.7, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(comparison_df['num_trees'], 0, comparison_df['improvement_pct'],
                     where=(comparison_df['improvement_pct'] > 0),
                     color='green', alpha=0.2, label='Improvement')
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Percentage Improvement Over Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot2_path = output_dir / 'comparison_improvement_pct.png'
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=150)
    print(f"Plot saved: {plot2_path}")
    plt.close()

    # Plot 3: Absolute improvement
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(comparison_df['num_trees'], comparison_df['improvement'],
           color='blue', alpha=0.6, width=1.0)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Absolute Improvement')
    ax.set_title('Absolute Improvement in Bounding Square Side Length')
    ax.grid(True, alpha=0.3, axis='y')

    plot3_path = output_dir / 'comparison_improvement_abs.png'
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=150)
    print(f"Plot saved: {plot3_path}")
    plt.close()

    # Plot 4: Cumulative improvement
    comparison_df['cumulative_improvement'] = comparison_df['improvement'].cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['num_trees'], comparison_df['cumulative_improvement'],
            color='purple', linewidth=2)
    ax.fill_between(comparison_df['num_trees'], 0,
                     comparison_df['cumulative_improvement'],
                     color='purple', alpha=0.2)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Cumulative Improvement')
    ax.set_title('Cumulative Score Improvement Over Baseline')
    ax.grid(True, alpha=0.3)

    plot4_path = output_dir / 'comparison_cumulative.png'
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=150)
    print(f"Plot saved: {plot4_path}")
    plt.close()


if __name__ == "__main__":
    compare_results()
