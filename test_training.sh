#!/bin/bash
# ConTree Training Tests - Verify optimal classification tree construction

echo "=== ConTree Training Functionality Tests ==="
echo ""

echo "Test 1: Basic tree construction (depth-3, bank dataset)"
./ConTree -file ../../datasets/bank.txt -max-depth 3
echo ""

echo "Test 2: Depth-2 (specialized solver)"
./ConTree -file ../../datasets/bank.txt -max-depth 2
echo ""

echo "Test 3: Depth-4 (perfect accuracy)"
./ConTree -file ../../datasets/bank.txt -max-depth 4
echo ""

echo "Test 4: Different dataset (raisin)"
./ConTree -file ../../datasets/raisin.txt -max-depth 2
echo ""

echo "Test 5: With approximation (max-gap=10)"
./ConTree -file ../../datasets/bank.txt -max-depth 3 -max-gap 10
echo ""

echo "Test 6: With gini sorting"
./ConTree -file ../../datasets/bank.txt -max-depth 3 -sort-features-gini-index 1
echo ""

echo "=== All tests completed successfully ==="
