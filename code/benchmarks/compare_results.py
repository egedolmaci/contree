#!/usr/bin/env python3
"""
Compare Google Benchmark JSON results and calculate speedup.

Usage:
    python compare_results.py baseline.json parallel.json

Output:
    - Prints results to console
    - Saves results to comparison_<baseline>_vs_<comparison>.txt
"""

import json
import sys
from pathlib import Path
from datetime import datetime


class TeeOutput:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_benchmark_results(filepath):
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_benchmarks(data):
    """Extract benchmark results into a dictionary keyed by benchmark name."""
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench['name']
        benchmarks[name] = {
            'real_time': bench['real_time'],
            'cpu_time': bench['cpu_time'],
            'iterations': bench['iterations']
        }
    return benchmarks


def calculate_speedup(baseline_time, comparison_time):
    """Calculate speedup factor."""
    if comparison_time == 0:
        return float('inf')
    return baseline_time / comparison_time


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.3f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    else:
        return f"{seconds:.3f} s"


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <baseline.json> <comparison.json>")
        sys.exit(1)

    baseline_file = Path(sys.argv[1])
    comparison_file = Path(sys.argv[2])

    if not baseline_file.exists():
        print(f"Error: Baseline file '{baseline_file}' not found")
        sys.exit(1)

    if not comparison_file.exists():
        print(f"Error: Comparison file '{comparison_file}' not found")
        sys.exit(1)

    # Generate output filename
    baseline_name = baseline_file.stem  # filename without extension
    comparison_name = comparison_file.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create comparisons directory if it doesn't exist
    comparisons_dir = baseline_file.parent.parent / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    output_file = comparisons_dir / f"comparison_{baseline_name}_vs_{comparison_name}.txt"

    # Set up output to both console and file
    tee = TeeOutput(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    # Print header with metadata
    print(f"Benchmark Comparison Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baseline: {baseline_file}")
    print(f"Comparison: {comparison_file}")
    print()

    # Load results
    print(f"Loading baseline: {baseline_file.name}")
    baseline_data = load_benchmark_results(baseline_file)
    baseline_benchmarks = extract_benchmarks(baseline_data)

    print(f"Loading comparison: {comparison_file.name}")
    comparison_data = load_benchmark_results(comparison_file)
    comparison_benchmarks = extract_benchmarks(comparison_data)

    # Dataset names mapping
    dataset_names = {
        0: "bank",
        1: "raisin",
        2: "wilt",
        3: "rice",
        4: "segment",
        5: "bidding",
        6: "fault",
        7: "page"
    }

    # Print header
    print("\n" + "=" * 100)
    print(f"Benchmark Comparison: {baseline_file.name} vs {comparison_file.name}")
    print("=" * 100)
    print()

    # Table header
    print(f"{'Benchmark':<25} {'Baseline Time':<15} {'Comparison Time':<15} {'Speedup':<12} {'Improvement':<12}")
    print("-" * 100)

    # Compare common benchmarks
    total_speedup = 0
    count = 0

    for name in sorted(baseline_benchmarks.keys()):
        if name in comparison_benchmarks:
            baseline = baseline_benchmarks[name]
            comparison = comparison_benchmarks[name]

            # Use CPU time for comparison (more stable than real time)
            baseline_time = baseline['cpu_time']
            comparison_time = comparison['cpu_time']

            speedup = calculate_speedup(baseline_time, comparison_time)
            improvement = (speedup - 1) * 100

            # Parse benchmark name to get dataset and depth
            # Format: BM_ConTree/dataset_idx/depth
            parts = name.split('/')
            if len(parts) == 3:
                dataset_idx = int(parts[1])
                depth = parts[2]
                dataset = dataset_names.get(dataset_idx, f"dataset{dataset_idx}")
                bench_name = f"{dataset}_d{depth}"
            else:
                bench_name = name

            print(f"{bench_name:<25} {format_time(baseline_time):<15} {format_time(comparison_time):<15} "
                  f"{speedup:>6.2f}x{'':<5} {improvement:>6.1f}%")

            total_speedup += speedup
            count += 1

    # Print summary
    print("-" * 100)
    if count > 0:
        avg_speedup = total_speedup / count
        avg_improvement = (avg_speedup - 1) * 100
        print(f"{'AVERAGE':<25} {'':<15} {'':<15} {avg_speedup:>6.2f}x{'':<5} {avg_improvement:>6.1f}%")
        print("=" * 100)
        print()

        # Additional statistics
        print("Summary:")
        print(f"  • Total benchmarks compared: {count}")
        print(f"  • Average speedup: {avg_speedup:.2f}x")
        print(f"  • Average improvement: {avg_improvement:.1f}%")

        if avg_speedup > 1:
            print(f"  • Overall: {comparison_file.name} is FASTER ✓")
        elif avg_speedup < 1:
            slowdown = 1 / avg_speedup
            print(f"  • Overall: {comparison_file.name} is SLOWER ({slowdown:.2f}x slowdown)")
        else:
            print(f"  • Overall: Performance is IDENTICAL")
    else:
        print("No matching benchmarks found!")

    print()

    # Close file output and restore stdout
    sys.stdout = original_stdout
    tee.close()

    # Print confirmation message
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
