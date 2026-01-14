#ifndef CACHE_H
#define CACHE_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "dataview.h"
#include "tree.h"

// 1. Define Entry and Shard FIRST
struct CacheEntry {
    int depth;
    std::shared_ptr<Tree> solution;
    CacheEntry() : depth(-1), solution(nullptr) {}
    CacheEntry(int d, std::shared_ptr<Tree> s) : depth(d), solution(s) {}
    bool is_set() const { return solution != nullptr; }
};

struct CacheShard {
    std::mutex mtx;
    std::unordered_map<DataviewBitset, CacheEntry> table;
};

// 2. Define Cache Class SECOND
class Cache {
public:
    static Cache global_cache;
    static constexpr int NUM_SHARDS = 64; 

    Cache() : use_caching(true) {}
    Cache(int max_depth, int num_instances);

    bool is_cached(const Dataview& data, int depth);
    void store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree);
    std::shared_ptr<Tree> retrieve(const Dataview& data, int depth);

private:
    bool use_caching;
    
    // Flattened structure: _cache[depth][shard_index]
    std::vector<std::vector<std::unique_ptr<CacheShard>>> _cache;
    
    inline int get_shard_idx(const DataviewBitset& bitset) const {
        return bitset.get_hash() & (NUM_SHARDS - 1); 
    }
};

#endif // CACHE_H