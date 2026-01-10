#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <chrono>
#include "stopwatch.h"
#include <cassert>
#include <stdexcept>
#include <iostream>

#define EPSILON 0.0000001f

#define PRINT_INTERMEDIARY_TIME_SOLUTIONS 0
extern std::chrono::high_resolution_clock::time_point starting_time;




#ifdef NDEBUG
#define RUNTIME_ASSERT(cond, msg) ((void)0)
#else
#define RUNTIME_ASSERT(cond, msg)                                             \
        do {                                                                      \
            if (!(cond)) {                                                        \
                std::cerr << "Assertion failed: " << #cond                        \
                          << "\nMessage: " << msg                                 \
                          << "\nFile: " << __FILE__                               \
                          << "\nLine: " << __LINE__ << std::endl;                 \
                std::terminate();                                                 \
            }                                                                     \
        } while (0)
#endif

struct Configuration {
    int max_depth;
    int max_gap;
    float max_gap_decay;
    bool print_logs;
    bool use_upper_bound;
    bool use_gpu_bruteforce{ false };
    bool serialize_gpu_calls{ false };
    bool defer_gpu_checks{ false };
    int max_thresholds_per_feature{ 128 };
    bool is_root{ false };
    bool sort_gini{ false };
    Stopwatch stopwatch;
    Configuration GetLeftSubtreeConfig() const;
    Configuration GetRightSubtreeConfig(int left_gap) const;
};

#endif // CONFIGURATION_H
