"""Monitor the progress of running algorithms."""
import time
from pathlib import Path


def monitor_progress():
    """Monitor progress of baseline and optimized runs."""
    output_dir = Path(__file__).parent.parent / 'output'
    baseline_log = output_dir / 'baseline_run.log'
    optimized_log = output_dir / 'optimized_run.log'

    print("Monitoring algorithm progress...")
    print("=" * 80)

    while True:
        print("\n" + "=" * 80)
        print(f"Status at {time.strftime('%H:%M:%S')}")
        print("=" * 80)

        # Check baseline
        if baseline_log.exists():
            with open(baseline_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_lines = [l for l in lines[-20:] if l.strip()]
                    print("\nBASELINE (last 5 lines):")
                    print("-" * 80)
                    for line in last_lines[-5:]:
                        print(line.rstrip())
        else:
            print("\nBASELINE: Not started yet")

        # Check optimized
        if optimized_log.exists():
            with open(optimized_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_lines = [l for l in lines[-20:] if l.strip()]
                    print("\nOPTIMIZED (last 5 lines):")
                    print("-" * 80)
                    for line in last_lines[-5:]:
                        print(line.rstrip())
        else:
            print("\nOPTIMIZED: Not started yet")

        # Check if both are done
        baseline_done = baseline_log.exists() and any('Total Score' in line for line in open(baseline_log))
        optimized_done = optimized_log.exists() and any('Total Score' in line for line in open(optimized_log))

        if baseline_done and optimized_done:
            print("\n" + "=" * 80)
            print("BOTH ALGORITHMS COMPLETED!")
            print("=" * 80)
            break

        time.sleep(10)


if __name__ == "__main__":
    monitor_progress()
