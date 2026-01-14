#!/bin/bash

CORES=$(nproc)

for ((i=0; i<CORES; i++)); do
    taskset -c $i ./build/ConTree -file ../datasets/avila.txt -max-depth 3 &
done

wait