#!/bin/bash
#
# Run benchmarks with different thread counts
#
# Usage:
#   ./run_thread_benchmark.sh [filter] [thread_counts]
#
# Examples:
#   ./run_thread_benchmark.sh                    # Run all benchmarks with 1,2,4,8 threads
#   ./run_thread_benchmark.sh "bank|raisin"      # Run only bank and raisin datasets
#   ./run_thread_benchmark.sh ".*" "1 4 8"       # Custom thread counts
#

set -e

# Configuration
BENCHMARK_EXE="../build/contree_benchmarks"
RESULTS_DIR="/results"
FILTER="${1:-.*}"  # Default: all benchmarks
THREADS="${2:-1 2 4 8}"  # Default thread counts

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenMP Thread Count Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Filter: $FILTER"
echo "Thread counts: $THREADS"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if benchmark exists
if [ ! -f "$BENCHMARK_EXE" ]; then
    echo -e "${RED}Error: $BENCHMARK_EXE not found${NC}"
    echo "Please build first: make contree_benchmarks"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run benchmarks for each thread count
for threads in $THREADS; do
    OUTPUT_FILE="$RESULTS_DIR/hybrid_${threads}threads.json"

    echo -e "${YELLOW}Running with $threads thread(s)...${NC}"

    export OMP_NUM_THREADS=$threads

    $BENCHMARK_EXE \
        --benchmark_filter="$FILTER" \
        --benchmark_out="$OUTPUT_FILE" \
        --benchmark_out_format=json

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Saved: $OUTPUT_FILE${NC}"
    else
        echo -e "${RED}✗ Failed with $threads threads${NC}"
        exit 1
    fi

    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All benchmarks completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Result files:"
for threads in $THREADS; do
    echo "  - $RESULTS_DIR/hybrid_${threads}threads.json"
done
echo ""
echo "To compare results:"
echo "  python3 compare_results.py results/anything.json results/anything.json"
