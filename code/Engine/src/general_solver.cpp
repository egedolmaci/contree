#include "general_solver.h"
#include <omp.h>
#ifdef USE_CUDA
#include "GPUBruteForceSolver.h"
#endif

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    if (current_optimal_decision_tree->misclassification_score == 0 || dataview.get_dataset_size() == 0) {
        return;
    }

    calculate_leaf_node(dataview.get_class_number(), dataview.get_dataset_size(), dataview.get_label_frequency(), current_optimal_decision_tree);

    if (solution_configuration.max_depth == 0) {
        return;
    }

    if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap || dataview.get_dataset_size() == 1) {
        return;
    }

#ifdef USE_CUDA
    if (solution_configuration.use_gpu_bruteforce && solution_configuration.max_depth <= 2) {
        if (solution_configuration.serialize_gpu_calls) {
            #pragma omp critical(gpu_call)
            {
                GPUBruteForceSolver::solve(dataview, solution_configuration, current_optimal_decision_tree);
            }
        } else {
            GPUBruteForceSolver::solve(dataview, solution_configuration, current_optimal_decision_tree);
        }
        return;
    }
#endif

    if (solution_configuration.max_depth == 2) {
        SpecializedSolver::create_optimal_decision_tree(dataview, solution_configuration, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));
        return;
    }

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

#ifdef USE_CUDA
    if (solution_configuration.use_gpu_bruteforce && solution_configuration.defer_gpu_checks) {
        GPUBruteForceSolver::RunDeferredChecks();
    }
#endif

    // Final early return check
    if (current_optimal_decision_tree->misclassification_score == 0 ||
        !solution_configuration.stopwatch.IsWithinTimeLimit()) {
        return;
    }
}

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {    
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) return;
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        // Lock-free IntervalsPruner access
        bool should_prune = interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score);

        if (should_prune) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
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

        std::shared_ptr<Tree> left_optimal_dt  = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);

        // Here firstly compute the bigger dataset since it might make computing the smaller dataset obsolete
        auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size() ) ? left_dataview : right_dataview;
        auto& larger_data  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_dataview : left_dataview;

        auto& smaller_optimal_dt = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? left_optimal_dt : right_optimal_dt;
        auto& larger_optimal_dt  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_optimal_dt : left_optimal_dt;

        int larger_ub = solution_configuration.use_upper_bound ? std::min(upper_bound, current_optimal_decision_tree->misclassification_score)
                                       : current_optimal_decision_tree->misclassification_score;

        statistics::total_number_of_general_solver_calls += 1;

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
            RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "Left tree should have non-negative misclassification score.");
            RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "Right tree should have non-negative misclassification score.");

            const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

            if (current_best_score < current_optimal_decision_tree->misclassification_score) {
                RUNTIME_ASSERT(left_optimal_dt->is_initialized(), "Left tree should be initialized.");
                RUNTIME_ASSERT(right_optimal_dt->is_initialized(), "Right tree should be initialized.");

                current_optimal_decision_tree->misclassification_score = current_best_score;
                current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

                if (current_best_score == 0) {
                    return;
                }

                if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root)  {
                    const auto stop = std::chrono::high_resolution_clock::now();
                    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                    std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
                }
            }
        } else {
            smaller_optimal_dt->misclassification_score = -1;
        }

        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score - current_optimal_decision_tree->misclassification_score;

        const auto bounds = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);
        int new_bound_left = bounds.first;
        int new_bound_right = bounds.second;

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
