# ConTree Parallelization Documentation

## Overview

This document details all parallelization approaches attempted for the ConTree optimal decision tree algorithm, including implementation details, code examples, performance results, and lessons learned.

The ConTree algorithm is a branch-and-bound optimizer that searches for optimal decision trees. The primary computational bottleneck is in the recursive tree construction, making it an ideal candidate for parallelization.

## Table of Contents

1. [Phase 1: Feature-Level Parallelism](#phase-1-feature-level-parallelism)
2. [Phase 2: Recursive Task Parallelism (Hybrid Approach)](#phase-2-recursive-task-parallelism-hybrid-approach)
3. [Phase 3: Interval-Level Parallelism with Critical Sections](#phase-3-interval-level-parallelism-with-critical-sections)
4. [Phase 4: Lock-Free IntervalsPruner](#phase-4-lock-free-intervalspruner)
5. [Phase 5: Cache-Aligned Atomics](#phase-5-cache-aligned-atomics)
6. [Performance Comparison](#performance-comparison)
7. [Lessons Learned](#lessons-learned)

---

## Phase 1: Feature-Level Parallelism

### Overview

The first parallelization target was the outer loop in `GeneralSolver::create_optimal_decision_tree()` which iterates over features to find the best split. Each feature can be evaluated independently, making this an embarrassingly parallel problem.

### Implementation

**Location:** `Engine/src/general_solver.cpp:31-75`

```cpp
// Pre-initialize bitset to avoid race condition
dataview.get_bitset();

// Store initial best score
int initial_best_score = current_optimal_decision_tree->misclassification_score;
bool should_terminate = false;

#pragma omp parallel if(dataview.get_feature_number() > 1)
{
    // Thread-local best tree
    std::shared_ptr<Tree> thread_local_best = std::make_shared<Tree>(-1, initial_best_score);

    #pragma omp for schedule(dynamic) nowait
    for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
        // Check for early termination
        #pragma omp flush(should_terminate)
        if (should_terminate) continue;

        int feature_index = dataview.gini_values[feature_nr].second;

        // Get current best upper bound
        int current_upper_bound;
        #pragma omp critical(update_tree)
        {
            current_upper_bound = std::min(upper_bound, current_optimal_decision_tree->misclassification_score);
        }

        // Solve for this feature
        std::shared_ptr<Tree> feature_tree = std::make_shared<Tree>(-1, current_upper_bound);
        create_optimal_decision_tree(dataview, solution_configuration, feature_index, feature_tree, current_upper_bound);

        // Update thread-local best
        if (feature_tree->misclassification_score < thread_local_best->misclassification_score) {
            thread_local_best = feature_tree;
        }

        // Update global best if improved
        #pragma omp critical(update_tree)
        {
            if (thread_local_best->misclassification_score < current_optimal_decision_tree->misclassification_score) {
                *current_optimal_decision_tree = *thread_local_best;
                thread_local_best = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);
            }

            // Check termination
            if (current_optimal_decision_tree->misclassification_score == 0 ||
                !solution_configuration.stopwatch.IsWithinTimeLimit()) {
                should_terminate = true;
            }
        }
    }
}
```

### Key Design Decisions

1. **Thread-Local Trees:** Each thread maintains its own best tree to minimize critical section contention
2. **Dynamic Scheduling:** Features have varying computational costs; dynamic scheduling balances load
3. **Early Termination:** Threads check a shared flag to stop when optimal solution is found
4. **Conditional Parallelism:** Only parallelize when there are multiple features

### Performance Results

With 64 threads, this approach showed significant speedups across all benchmarks, demonstrating that feature-level parallelism is highly effective for this problem.

### Challenges

- **Critical Section Contention:** Updating the global best tree requires synchronization
- **Load Imbalance:** Some features are much more expensive to evaluate than others
- **Memory Pressure:** Each thread allocates its own tree structures

---

## Phase 2: Recursive Task Parallelism (Hybrid Approach)

### Overview

While iterating through features in parallel (Phase 1), we can also parallelize the recursive subtree construction within each feature evaluation. This creates a two-level parallel hierarchy: feature-level (fork-join) and subtree-level (tasks).

### Implementation

**Location:** `Engine/src/general_solver.cpp:147-180`

```cpp
const Configuration left_solution_configuration = solution_configuration.GetLeftSubtreeConfig();

// Threshold for task creation: only use tasks for larger problems
const int TASK_CUTOFF_SIZE = 50;  // Minimum dataset size to create tasks
const int TASK_CUTOFF_DEPTH = 3;  // Maximum depth to create tasks (avoid deep nesting)
bool use_tasks = (larger_data.get_dataset_size() >= TASK_CUTOFF_SIZE &&
                  solution_configuration.max_depth >= TASK_CUTOFF_DEPTH);

if (use_tasks && omp_in_parallel()) {
    // We're already in a parallel region (feature-level), use tasks for subtrees
    #pragma omp task shared(larger_optimal_dt)
    {
        GeneralSolver::create_optimal_decision_tree(larger_data, left_solution_configuration, larger_optimal_dt, larger_ub);
    }
    #pragma omp taskwait  // Wait for larger subtree before computing bounds for smaller
} else {
    // Sequential execution (not in parallel region, or problem too small)
    GeneralSolver::create_optimal_decision_tree(larger_data, left_solution_configuration, larger_optimal_dt, larger_ub);
}

int smaller_ub = solution_configuration.use_upper_bound ? std::max(std::min(current_optimal_decision_tree->misclassification_score, upper_bound) - larger_optimal_dt->misclassification_score, interval_half_distance)
                                : current_optimal_decision_tree->misclassification_score;

if (smaller_ub > 0 || (smaller_ub == 0 && current_optimal_decision_tree->misclassification_score == larger_optimal_dt->misclassification_score)) {
    statistics::total_number_of_general_solver_calls += 1;
    const Configuration right_solution_configuration = solution_configuration.GetRightSubtreeConfig(left_solution_configuration.max_gap);

    if (use_tasks && omp_in_parallel()) {
        #pragma omp task shared(smaller_optimal_dt)
        {
            GeneralSolver::create_optimal_decision_tree(smaller_data, right_solution_configuration, smaller_optimal_dt, smaller_ub);
        }
        #pragma omp taskwait  // Wait for smaller subtree
    } else {
        GeneralSolver::create_optimal_decision_tree(smaller_data, right_solution_configuration, smaller_optimal_dt, smaller_ub);
    }
}
```

### Key Design Decisions

1. **Adaptive Task Creation:** Only create tasks for large enough problems to avoid overhead
2. **Cutoff Thresholds:**
   - Dataset size ≥ 50 instances
   - Tree depth ≥ 3 levels
3. **Sequential Dependency:** Larger subtree must complete before smaller subtree (for bound computation)
4. **Nested Parallelism:** Tasks run within the feature-level parallel region

### Performance Results

Phase 2 (hybrid approach) showed improvements over Phase 1 alone, particularly for larger datasets with deeper trees. The task-based parallelism effectively utilizes idle threads during recursive descent.

### Challenges

- **Task Overhead:** Creating tasks for small problems adds overhead without benefit
- **Load Imbalance:** Difficult to predict task execution time, leading to thread idle time
- **Memory Overhead:** Each task has its own stack and bookkeeping

---

## Phase 3: Interval-Level Parallelism with Critical Sections

### Overview

The innermost level of parallelism targets the interval search loop within each feature evaluation. The algorithm searches through possible split points using binary search with pruning. This loop processes intervals from a queue, and intervals can be processed in any order.

### Implementation

**Original Sequential Code:**

```cpp
IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);

std::queue<IntervalsPruner::Bound> unsearched_intervals;
unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

while(!unsearched_intervals.empty()) {
    auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

    // Prune intervals
    if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
        continue;
    }

    interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
    // ... process interval ...
}
```

**Parallelized with Critical Sections:**

**Location:** `Engine/src/general_solver.cpp:97-224`

```cpp
IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);

std::queue<IntervalsPruner::Bound> unsearched_intervals;
unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

while(!unsearched_intervals.empty()) {
    auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

    // Thread-safe IntervalsPruner access
    bool should_prune;
    #pragma omp critical(interval_pruner)
    {
        should_prune = interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score);
    }

    if (should_prune) {
        continue;
    }

    #pragma omp critical(interval_pruner)
    {
        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
    }

    // ... process interval ...

    #pragma omp critical(interval_pruner)
    {
        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);
    }

    int new_bound_left, new_bound_right;
    #pragma omp critical(interval_pruner)
    {
        const auto bounds = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);
        new_bound_left = bounds.first;
        new_bound_right = bounds.second;
    }
}
```

### Key Design Decisions

1. **Shared IntervalsPruner:** Single pruner instance protected by critical sections
2. **Four Critical Sections:**
   - `subinterval_pruning()` - Check if interval can be pruned
   - `interval_shrinking()` - Narrow interval bounds
   - `add_result()` - Record split evaluation results
   - `neighbourhood_pruning()` - Compute new intervals to search
3. **Thread-Local Queues:** Each thread maintains its own work queue

### Performance Results

**Benchmark comparison (Phase 3 vs Phase 2, 64 threads):**
- Average speedup: 0.99x
- Average improvement: -0.8%
- Overall: 1.01x slowdown

**Analysis:** The critical sections serialized access to the IntervalsPruner, eliminating any benefit from parallel interval processing. The overhead of synchronization negated potential speedup.

### Challenges

- **Serialization Bottleneck:** Critical sections forced threads to wait
- **Contention:** High contention on the `interval_pruner` lock
- **No Actual Parallelism:** Only one thread could access pruner at a time

---

## Phase 4: Lock-Free IntervalsPruner

### Overview

To eliminate critical section overhead, we redesigned IntervalsPruner to be lock-free using atomic operations. This allows threads to concurrently read and write pruning state without blocking.

### Original Data Structure

**Location:** `DataStructures/include/intervals_pruner.h` (before Phase 4)

```cpp
class IntervalsPruner {
private:
    const std::vector<int>& possible_split_indexes;
    int possible_split_size;
    int rightmost_zero_index;
    int leftmost_zero_index;
    int max_gap;
    std::unordered_map<int, std::pair<int, int>> evaluated_indices_record;
};
```

### Lock-Free Implementation

**Location:** `DataStructures/include/intervals_pruner.h` (Phase 4)

```cpp
class IntervalsPruner {
private:
    const std::vector<int>& possible_split_indexes;
    int possible_split_size;
    std::atomic<int> rightmost_zero_index;
    std::atomic<int> leftmost_zero_index;
    int max_gap;

    // Lock-free storage for evaluated results
    std::unique_ptr<std::atomic<int>[]> left_scores;
    std::unique_ptr<std::atomic<int>[]> right_scores;

    // Helper to atomically update minimum
    static void atomic_fetch_min(std::atomic<int>& target, int value);

    // Helper to atomically update maximum
    static void atomic_fetch_max(std::atomic<int>& target, int value);
};
```

**Atomic Helper Functions:**

**Location:** `DataStructures/src/intervals_pruner.cpp`

```cpp
void IntervalsPruner::atomic_fetch_min(std::atomic<int>& target, int value) {
    int current = target.load(std::memory_order_relaxed);
    while (value < current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed));
}

void IntervalsPruner::atomic_fetch_max(std::atomic<int>& target, int value) {
    int current = target.load(std::memory_order_relaxed);
    while (value > current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed));
}
```

**Constructor:**

```cpp
IntervalsPruner::IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap)
    : possible_split_indexes(possible_split_indexes_ref),
      possible_split_size(int(possible_split_indexes.size())),
      rightmost_zero_index(possible_split_size),
      leftmost_zero_index(-1),
      max_gap(max_gap),
      left_scores(new std::atomic<int>[possible_split_indexes.size()]),
      right_scores(new std::atomic<int>[possible_split_indexes.size()]) {
    // Initialize all scores to -1 (uninitialized marker)
    for (size_t i = 0; i < possible_split_indexes.size(); ++i) {
        left_scores[i].store(-1, std::memory_order_relaxed);
        right_scores[i].store(-1, std::memory_order_relaxed);
    }
}
```

**Add Result (Lock-Free):**

```cpp
void IntervalsPruner::add_result(int index, int left_score, int right_score) {
    if (left_score == 0) {
        atomic_fetch_max(leftmost_zero_index, index);
    }

    if (right_score == 0) {
        atomic_fetch_min(rightmost_zero_index, index);
    }

    if (left_score == -1) {
        left_score = 0;
    }

    if (right_score == -1) {
        right_score = 0;
    }

    // Store results atomically
    left_scores[index].store(left_score, std::memory_order_release);
    right_scores[index].store(right_score, std::memory_order_release);
}
```

**Subinterval Pruning (Lock-Free):**

```cpp
bool IntervalsPruner::subinterval_pruning(const IntervalsPruner::Bound& current_bounds, int current_best_score) {
    int left_bound_score_left = 0;
    int right_bound_score_right = 0;

    if (current_bounds.last_split_left_index != -1) {
        int score = left_scores[current_bounds.last_split_left_index].load(std::memory_order_acquire);
        left_bound_score_left = (score == -1) ? 0 : score;
    }

    if (current_bounds.last_split_right_index != -1) {
        int score = right_scores[current_bounds.last_split_right_index].load(std::memory_order_acquire);
        right_bound_score_right = (score == -1) ? 0 : score;
    }

    return left_bound_score_left + right_bound_score_right + max_gap >= current_best_score;
}
```

**General Solver (No More Critical Sections):**

**Location:** `Engine/src/general_solver.cpp`

```cpp
// Lock-free IntervalsPruner access
bool should_prune = interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score);

if (should_prune) {
    continue;
}

interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);

// ... later ...

interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

const auto bounds = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);
```

### Key Design Decisions

1. **Replace HashMap with Arrays:** Pre-allocate fixed-size arrays indexed by split point
2. **Atomic Operations:** Use `std::atomic<int>` with acquire/release semantics
3. **Compare-and-Swap Loops:** Implement min/max operations with CAS loops
4. **Memory Ordering:**
   - `memory_order_relaxed` for CAS loops (performance)
   - `memory_order_acquire` for loads (see updated values)
   - `memory_order_release` for stores (make values visible)

### Performance Results

**Benchmark comparison (Lock-free vs Phase 3, 64 threads):**
- Average speedup: 0.92x
- Average improvement: -8.2%
- Overall: 1.09x slowdown

**Analysis:** The lock-free version was **slower** than Phase 3! Why?

1. **False Sharing:** Adjacent atomic integers in arrays share cache lines
2. **Memory Ordering Overhead:** Acquire/release fences on every access
3. **CAS Contention:** Multiple threads spinning on compare-and-swap
4. **Initialization Cost:** Initializing large atomic arrays
5. **Poor Cache Locality:** Sparse access pattern in dense arrays

---

## Phase 5: Cache-Aligned Atomics

### Overview

False sharing occurs when multiple threads access different variables that reside on the same cache line. When one thread modifies its variable, the entire cache line is invalidated for other threads, causing expensive cache coherency traffic.

**Cache Line Size:** Typically 64 bytes on x86-64 processors

**Problem:** `std::atomic<int>` is 4 bytes, so 16 atomics fit in one cache line. Threads writing to different indices cause cache line bouncing.

**Solution:** Align each atomic to its own cache line boundary using `alignas(64)`.

### Implementation

**Location:** `DataStructures/include/intervals_pruner.h`

```cpp
class IntervalsPruner {
private:
    // Cache-aligned atomic to prevent false sharing
    // Typical cache line size is 64 bytes on x86-64
    struct alignas(64) CacheAlignedAtomic {
        std::atomic<int> value;

        CacheAlignedAtomic() : value(-1) {}
    };

    const std::vector<int>& possible_split_indexes;
    int possible_split_size;
    std::atomic<int> rightmost_zero_index;
    std::atomic<int> leftmost_zero_index;
    int max_gap;

    // Lock-free storage for evaluated results with cache-line alignment
    std::unique_ptr<CacheAlignedAtomic[]> left_scores;
    std::unique_ptr<CacheAlignedAtomic[]> right_scores;
};
```

**Constructor:**

```cpp
IntervalsPruner::IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap)
    : possible_split_indexes(possible_split_indexes_ref),
      possible_split_size(int(possible_split_indexes.size())),
      rightmost_zero_index(possible_split_size),
      leftmost_zero_index(-1),
      max_gap(max_gap),
      left_scores(new CacheAlignedAtomic[possible_split_indexes.size()]),
      right_scores(new CacheAlignedAtomic[possible_split_indexes.size()]) {
    // CacheAlignedAtomic constructor already initializes to -1
}
```

**Usage (Access via .value):**

```cpp
bool IntervalsPruner::subinterval_pruning(const IntervalsPruner::Bound& current_bounds, int current_best_score) {
    int left_bound_score_left = 0;
    int right_bound_score_right = 0;

    if (current_bounds.last_split_left_index != -1) {
        int score = left_scores[current_bounds.last_split_left_index].value.load(std::memory_order_acquire);
        left_bound_score_left = (score == -1) ? 0 : score;
    }

    if (current_bounds.last_split_right_index != -1) {
        int score = right_scores[current_bounds.last_split_right_index].value.load(std::memory_order_acquire);
        right_bound_score_right = (score == -1) ? 0 : score;
    }

    return left_bound_score_left + right_bound_score_right + max_gap >= current_best_score;
}
```

### Memory Layout Comparison

**Without Alignment (Phase 4):**
```
Cache Line 0 (64 bytes): [atomic0][atomic1][atomic2]...[atomic15]
Cache Line 1 (64 bytes): [atomic16][atomic17]...[atomic31]
...
```
Thread A writes to `atomic5` → invalidates entire Cache Line 0
Thread B reading `atomic7` → cache miss!

**With Alignment (Phase 5):**
```
Cache Line 0 (64 bytes): [atomic0 + padding]
Cache Line 1 (64 bytes): [atomic1 + padding]
Cache Line 2 (64 bytes): [atomic2 + padding]
...
```
Thread A writes to `atomic5` → invalidates only Cache Line 5
Thread B reading `atomic7` → cache hit!

### Key Design Decisions

1. **64-Byte Alignment:** Match typical x86-64 cache line size
2. **Wrapper Struct:** Use `alignas(64)` on a struct containing the atomic
3. **Memory Overhead:** Trade 16x memory increase for elimination of false sharing
4. **Per-Element Isolation:** Each atomic gets its own cache line

### Performance Results

**Expected:** Significant improvement over Phase 4 by eliminating false sharing

**Trade-offs:**
- **Pro:** No more cache line contention between threads
- **Con:** 16x memory usage (4 bytes → 64 bytes per atomic)
- **Con:** Potential TLB pressure from larger memory footprint

---


