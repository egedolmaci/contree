#include "general_solver.h"
#include <omp.h>
#include <atomic>
#include <algorithm> // Required for std::min
#include <climits>   // Required for INT_MAX

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    if (current_optimal_decision_tree->misclassification_score == 0 || dataview.get_dataset_size() == 0) return;

    // Cache lookup
    if (Cache::global_cache.is_cached(dataview, solution_configuration.max_depth)) {
        current_optimal_decision_tree = Cache::global_cache.retrieve(dataview, solution_configuration.max_depth);
        return;
    }

    calculate_leaf_node(dataview.get_class_number(), dataview.get_dataset_size(), dataview.get_label_frequency(), current_optimal_decision_tree);

    if (solution_configuration.max_depth == 0) return;
    if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap || dataview.get_dataset_size() == 1) return;

    // Handoff to SpecializedSolver for depth 2
    if (solution_configuration.max_depth == 2) {
        SpecializedSolver::create_optimal_decision_tree(dataview, solution_configuration, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));
        return;
    }

    // Local atomic to synchronize threads ONLY for this specific node
    std::atomic<int> node_best_score(current_optimal_decision_tree->misclassification_score);

    int num_features = dataview.get_feature_number();
    
    #pragma omp parallel for schedule(dynamic) if(dataview.get_dataset_size() > 500)
    for (int feature_nr = 0; feature_nr < num_features; feature_nr++) {
        // Early exit if optimal found
        int current_best = node_best_score.load(std::memory_order_relaxed);
        if (current_best == 0) continue;
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) continue;

        int feature_index = dataview.gini_values[feature_nr].second;

        // [FIX] Initialize local tree with INT_MAX so any valid solution overwrites it.
        // We rely on 'upper_bound' (passed below) to handle the pruning constraints.
        std::shared_ptr<Tree> local_tree = std::make_shared<Tree>(-1, INT_MAX);

        create_optimal_decision_tree(dataview, solution_configuration, feature_index, local_tree, std::min(upper_bound, current_best), node_best_score);

        // [FIX] Only update if local_tree was actually populated (score < INT_MAX)
        if (local_tree->misclassification_score < INT_MAX && 
            local_tree->misclassification_score <= node_best_score.load(std::memory_order_relaxed)) {
            #pragma omp critical(update_tree)
            {
                if (local_tree->misclassification_score < current_optimal_decision_tree->misclassification_score) {
                    current_optimal_decision_tree = local_tree;
                    node_best_score.store(local_tree->misclassification_score, std::memory_order_relaxed);
                }
            }
        }
    }

    if (current_optimal_decision_tree->misclassification_score <= upper_bound) {
        Cache::global_cache.store(dataview, solution_configuration.max_depth, current_optimal_decision_tree);
    }
}

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound, std::atomic<int>& parent_node_best_score) {    
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        // Pruning Logic:
        // 1. Get the best score found by ANY sibling thread.
        int sibling_best = parent_node_best_score.load(std::memory_order_relaxed);
        
        // 2. Use the tighter of (My Current Best) vs (Sibling Best) for pruning.
        // Note: current_optimal_decision_tree->misclassification_score starts at INT_MAX, so this safely picks sibling_best initially.
        int pruning_bound = std::min(current_optimal_decision_tree->misclassification_score, sibling_best);

        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) return;
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, pruning_bound)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, pruning_bound);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if (left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const int interval_half_distance = std::max(split_point - possible_split_indices[left], possible_split_indices[right] - split_point);

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  
        const int split_unique_value_index = current_feature[split_point].unique_value_index;

        Dataview left_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview right_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview::split_data_points(dataview, feature_index, split_point, split_unique_value_index, left_dataview, right_dataview, solution_configuration.max_depth);

        // [FIX] Initialize children with INT_MAX.
        // This ensures that if the recursive call finds a tree with score == pruning_bound,
        // it is strictly less than INT_MAX, so it WILL update the pointer.
        std::shared_ptr<Tree> left_optimal_dt  = std::make_shared<Tree>(-1, INT_MAX);
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>(-1, INT_MAX);

        // TASK-BASED PARALLELISM: Compute left and right subtrees in parallel
        bool use_tasks = (left_dataview.get_dataset_size() > 200 && right_dataview.get_dataset_size() > 200);
        
        statistics::increment_gen();
        
        const Configuration left_solution_configuration = solution_configuration.GetLeftSubtreeConfig();
        const Configuration right_solution_configuration = solution_configuration.GetRightSubtreeConfig(left_solution_configuration.max_gap);
        
        // Pass pruning_bound as the upper_bound constraint for recursion
        int left_ub = solution_configuration.use_upper_bound ? std::min(upper_bound, pruning_bound)
                                       : pruning_bound;
        
        if (use_tasks) {
            #pragma omp task shared(left_optimal_dt) firstprivate(left_dataview, left_solution_configuration, left_ub)
            {
                GeneralSolver::create_optimal_decision_tree(left_dataview, left_solution_configuration, left_optimal_dt, left_ub);
            }
            
            #pragma omp task shared(right_optimal_dt) firstprivate(right_dataview, right_solution_configuration, left_ub)
            {
                GeneralSolver::create_optimal_decision_tree(right_dataview, right_solution_configuration, right_optimal_dt, left_ub);
            }
            
            #pragma omp taskwait
        } else {
            // Sequential for small subtrees (less overhead)
            auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size() ) ? left_dataview : right_dataview;
            auto& larger_data  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_dataview : left_dataview;

            auto& smaller_optimal_dt = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? left_optimal_dt : right_optimal_dt;
            auto& larger_optimal_dt  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_optimal_dt : left_optimal_dt;

            GeneralSolver::create_optimal_decision_tree(larger_data, left_solution_configuration, larger_optimal_dt, left_ub);

            // Dynamically calculate remaining budget for the second tree
            // If the first tree failed (still INT_MAX), we can't calculate a budget, so we skip the second.
            if (larger_optimal_dt->misclassification_score < INT_MAX) {
                 int smaller_ub = solution_configuration.use_upper_bound ? std::max(std::min(pruning_bound, upper_bound) - larger_optimal_dt->misclassification_score, interval_half_distance) 
                                            : pruning_bound;

                 if (smaller_ub > 0 || (smaller_ub == 0 && pruning_bound == larger_optimal_dt->misclassification_score)) {
                     statistics::increment_gen();
                     GeneralSolver::create_optimal_decision_tree(smaller_data, right_solution_configuration, smaller_optimal_dt, smaller_ub);
                 }
            }
        }

        // Check if either child failed to find a valid solution (still INT_MAX) or is uninitialized
        if (left_optimal_dt->misclassification_score >= INT_MAX || right_optimal_dt->misclassification_score >= INT_MAX ||
            !left_optimal_dt->is_initialized() || !right_optimal_dt->is_initialized()) {
            interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);
            continue;
        }

        RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "Right tree should have non-negative misclassification score.");

        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {
            // Double check initialization before assignment
            RUNTIME_ASSERT(left_optimal_dt->is_initialized(), "Left tree should be initialized.");
            RUNTIME_ASSERT(right_optimal_dt->is_initialized(), "Right tree should be initialized.");

            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);
            
            // Update the passed atomic to signal other threads working on this node
            int old_node_best = parent_node_best_score.load(std::memory_order_relaxed);
            while (current_best_score < old_node_best && 
                   !parent_node_best_score.compare_exchange_weak(old_node_best, current_best_score, std::memory_order_relaxed));

            if (current_best_score == 0) {
                return;
            }

            if (solution_configuration.print_logs && solution_configuration.is_root)  {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }
        }
        
        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}

void GeneralSolver::calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree) {
    int best_classification_score = -1;
    int best_classification_label = -1;

    for (int label = 0; label < class_number; label++) {
        if (label_frequency[label] > best_classification_score) {
            best_classification_score = label_frequency[label];
            best_classification_label = label;
        }
    }

    const int best_misclassification_score = instance_number - best_classification_score;

    if (best_misclassification_score < current_optimal_decision_tree->misclassification_score) {
        RUNTIME_ASSERT(best_classification_label != -1, "Cannot assign negative leaf label.");
        current_optimal_decision_tree->make_leaf(best_classification_label, best_misclassification_score);
    }
}