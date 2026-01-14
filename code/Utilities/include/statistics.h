#ifndef STATISTICS_H
#define STATISTICS_H

#include <iostream>
#include <vector>
#include <omp.h>

namespace statistics {
    struct CacheLineCounter {
        alignas(64) unsigned long long value;
    };
    
    // Arrays of padded counters to prevent false sharing
    inline CacheLineCounter gen_calls[128];
    inline CacheLineCounter spec_calls[128];
    inline CacheLineCounter cache_hits[128];

    inline void increment_gen() { gen_calls[omp_get_thread_num()].value++; }
    inline void increment_spec() { spec_calls[omp_get_thread_num()].value++; }
    inline void increment_cache() { cache_hits[omp_get_thread_num()].value++; }

    void print_statistics();
}

#endif