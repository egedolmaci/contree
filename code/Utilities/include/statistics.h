#ifndef STATISTICS_H
#define STATISTICS_H

#include <atomic>

class statistics {
public:
    static std::atomic<unsigned long long> total_number_of_specialized_solver_calls;
    static std::atomic<unsigned long long> total_number_of_general_solver_calls;

    static bool should_print;

    static void print_statistics();
};

#endif // STATISTICS_H


