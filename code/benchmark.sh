#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/../datasets"
CON_TREE_BIN="$SCRIPT_DIR/../build/ConTree"
OUTPUT_FILE="final_results_openmp.csv"

MODE_NAME="CPU"

DATASETS=(
    "avila.txt"
    "bank.txt"
    "bean.txt"
    "bidding.txt"
    "eeg.txt"
    "fault.txt"
    "htru.txt"
    "magic.txt"
    "occupancy.txt"
    "page.txt"
    "raisin.txt"
    "rice.txt"
    "room.txt"
    "segment.txt"
    "skin.txt"
    "wilt.txt"
)

DEPTHS=(2 3 4 5)
THREADS=(1 2 3 4 5 6 7 8 9 10 15 30 60)


for depth in "${DEPTHS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for threads in "${THREADS[@]}"; do

            if grep -q "^$dataset,$depth,$MODE_NAME,$threads," "$OUTPUT_FILE"; then
                echo "SKIP (already done)"
                continue
            fi

            filepath="$DATASET_DIR/$dataset"

            if [ ! -f "$filepath" ]; then
                echo "WARNING: $filepath not found"
                continue
            fi

            echo -n "Running: $dataset depth=$depth threads=$threads ... "

            output=$(OMP_NUM_THREADS="$threads" "$CON_TREE_BIN" \
                -file "$filepath" \
                -max-depth "$depth" 2>&1)

            time_val=$(echo "$output" | grep "Average time taken" | awk '{print $9}')
            acc_val=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
            score_val=$(echo "$output" | grep "Misclassification score:" | awk '{print $3}')
            tree_val=$(echo "$output" | grep "Optimal tree:" | sed 's/Optimal tree: //')

            if [ -z "$time_val" ]; then
                echo "FAILED"
                echo "$output"
                continue
            fi

            echo "DONE ($time_val s)"

            echo "$dataset,$depth,$MODE_NAME,$threads,$time_val,$acc_val,$score_val,\"$tree_val\"" \
                >> "$OUTPUT_FILE"

        done
    done
done