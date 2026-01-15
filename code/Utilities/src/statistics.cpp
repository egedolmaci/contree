#include "statistics.h"

void statistics::print_statistics() {
    unsigned long long total_spec = 0;
    unsigned long long total_gen = 0;
    unsigned long long total_hits = 0;

    for (int i = 0; i < 128; ++i) {
        total_spec += spec_calls[i].value;
        total_gen += gen_calls[i].value;
        total_hits += cache_hits[i].value;
    }

    std::cout << "Total number of specialized solver calls: " << total_spec << std::endl;
    std::cout << "Total number of general solver calls: " << total_gen << std::endl;
    std::cout << "Total number cache hits: " << total_hits << std::endl;
}