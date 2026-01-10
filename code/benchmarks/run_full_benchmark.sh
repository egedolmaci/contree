#!/bin/bash

OUTPUT_FILE="benchmark_full.csv"

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
MODES=(0 1) # 0 = CPU, 1 = GPU

echo "Dataset,Depth,Mode,Time(s),Accuracy,Misclassification,TreeStructure" > "$OUTPUT_FILE"

echo "========================================================"
echo "Full Benchmark (Tree Capture) starting... Output: $OUTPUT_FILE"
echo "========================================================"

for depth in "${DEPTHS[@]}"; do
    echo ""
    echo "#############################################"
    echo "### PROCESSING DEPTH: $depth ###"
    echo "#############################################"

    for dataset in "${DATASETS[@]}"; do
        filepath="../../datasets/$dataset"

        if [ ! -f "$filepath" ]; then
            echo "WARNING: $filepath not found, skipping..."
            continue
        fi

        for mode in "${MODES[@]}"; do
            if [ "$mode" -eq 0 ]; then
                mode_name="CPU"
            else
                mode_name="GPU"
            fi

            echo -n "Running: $dataset (Depth $depth) [$mode_name]... "

            output=$(../../build/ConTree -file "$filepath" -max-depth "$depth" -use-gpu-bruteforce "$mode" 2>&1)

            time_val=$(echo "$output" | grep "Average time taken" | awk '{print $9}')
            acc_val=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
            score_val=$(echo "$output" | grep "Misclassification score:" | awk '{print $3}')
            tree_val=$(echo "$output" | grep "Optimal tree:" | sed 's/Optimal tree: //')

            if [ -z "$time_val" ]; then
                time_val="ERROR"
                acc_val="ERROR"
                score_val="ERROR"
                tree_val="ERROR"
                echo "FAILED!"
            else
                echo "DONE ($time_val s)"
            fi

            echo "$dataset,$depth,$mode_name,$time_val,$acc_val,$score_val,\"$tree_val\"" >> "$OUTPUT_FILE"

            sleep 0.5
        done
    done
done

echo ""
echo "========================================================"
echo "Benchmark complete!"
echo "Results: $OUTPUT_FILE"
