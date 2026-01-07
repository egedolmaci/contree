# Benchmark Comparison Tools

## Quick Start

### 1. Run Benchmarks with Different Thread Counts

```bash
# Run serial (1 thread)
cd build
OMP_NUM_THREADS=1 ./contree_benchmarks --benchmark_out=../benchmarks/results/serial.json --benchmark_out_format=json

# Run parallel with 4 threads
OMP_NUM_THREADS=4 ./contree_benchmarks --benchmark_out=../benchmarks/results/parallel_4threads.json --benchmark_out_format=json

# Run parallel with 8 threads
OMP_NUM_THREADS=8 ./contree_benchmarks --benchmark_out=../benchmarks/results/parallel_8threads.json --benchmark_out_format=json
```

### 2. Compare Results

```bash
cd benchmarks
python3 compare_results.py results/baseline.json results/feature4.json
```

## compare_results.py

Compares two Google Benchmark JSON files and calculates speedup.

**Usage:**
```bash
python3 compare_results.py <baseline.json> <comparison.json>
```

**Output:**
- Prints results to console
- **Automatically saves results** to `comparisons/comparison_<baseline>_vs_<comparison>.txt`
- Creates `comparisons/` directory automatically if it doesn't exist
- Detailed table showing baseline time, comparison time, speedup, and improvement percentage for each benchmark
- Average speedup across all benchmarks
- Summary statistics

**Example:**
```bash
$ python3 compare_results.py results/baseline.json results/parallel.json

====================================================================================================
Benchmark Comparison: baseline.json vs parallel.json
====================================================================================================

Benchmark                 Baseline Time   Comparison Time Speedup      Improvement
----------------------------------------------------------------------------------------------------
bank_d3                   72.000 ms       40.000 ms         1.80x        80.0%
bank_d4                   2.500 s         1.200 s           2.08x       108.3%
...
----------------------------------------------------------------------------------------------------
AVERAGE                                                     1.94x        94.0%
====================================================================================================

Summary:
  • Total benchmarks compared: 16
  • Average speedup: 1.94x
  • Average improvement: 94.0%
  • Overall: parallel.json is FASTER ✓
```

## Dataset Information

Datasets are ordered by size (small to medium):

| Index | Dataset | Size    | Features |
|-------|---------|---------|----------|
| 0     | bank    | 108 KB  | 16       |
| 1     | raisin  | 123 KB  | 7        |
| 2     | wilt    | 486 KB  | 5        |
| 3     | rice    | 518 KB  | 7        |
| 4     | segment | 754 KB  | 19       |
| 5     | bidding | 861 KB  | 8        |
| 6     | fault   | 938 KB  | 27       |
| 7     | page    | 1.1 MB  | 10       |

## Benchmark Naming

Benchmarks follow the format: `BM_ConTree/dataset_idx/depth`

Examples:
- `BM_ConTree/0/3` → bank dataset, depth 3
- `BM_ConTree/4/4` → segment dataset, depth 4

## Tips

1. **Run benchmarks in Release mode** for accurate timing:
   ```bash
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   ```

2. **Filter specific benchmarks:**
   ```bash
   ./contree_benchmarks --benchmark_filter="BM_ConTree/0/.*"  # Only bank dataset
   ./contree_benchmarks --benchmark_filter=".*/3"             # Only depth 3
   ```

3. **Set fixed thread count:**
   ```bash
   export OMP_NUM_THREADS=4
   ./contree_benchmarks --benchmark_out=results/4threads.json --benchmark_out_format=json
   ```

4. **Compare serial vs parallel:**
   ```bash
   # Serial
   OMP_NUM_THREADS=1 ./contree_benchmarks --benchmark_out=../benchmarks/results/serial.json --benchmark_out_format=json

   # Parallel
   OMP_NUM_THREADS=4 ./contree_benchmarks --benchmark_out=../benchmarks/results/parallel.json --benchmark_out_format=json

   # Compare
   cd ../benchmarks
   python3 compare_results.py results/serial.json results/parallel.json
   ```

## Understanding Results

- **Speedup > 1.0**: Comparison is faster (good!)
- **Speedup < 1.0**: Comparison is slower (needs investigation)
- **Speedup = 1.0**: Performance is identical

**Improvement Percentage** = (Speedup - 1) × 100%
- Positive: faster
- Negative: slower
