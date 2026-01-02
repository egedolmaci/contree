#include <iostream>

#include "statistics.h"

std::atomic<unsigned long long> statistics::total_number_of_specialized_solver_calls{0};
std::atomic<unsigned long long> statistics::total_number_of_general_solver_calls{0};

bool statistics::should_print = false;

void statistics::print_statistics() {
    std::cout << "Total number of specialized solver calls: " << total_number_of_specialized_solver_calls << std::endl;
    std::cout << "Total number of general solver calls: " << total_number_of_general_solver_calls << std::endl;
}
