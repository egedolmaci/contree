#include "cache.h"
#include "statistics.h"

Cache Cache::global_cache = Cache();

Cache::Cache(int max_depth, int num_instances) : use_caching(true) {
    // Resize for Depth + 1
    _cache.resize(max_depth + 1);
    
    for (int d = 0; d <= max_depth; ++d) {
        // Create 64 shards for this depth
        for (int i = 0; i < NUM_SHARDS; ++i) {
            _cache[d].push_back(std::make_unique<CacheShard>());
        }
    }
}

bool Cache::is_cached(const Dataview& data, int depth) {
    if (!use_caching) return false;

    const auto& bitset = data.get_bitset();
    if (!bitset.is_hash_set()) const_cast<DataviewBitset&>(bitset).set_hash(std::hash<DataviewBitset>()(bitset));

    int shard_idx = get_shard_idx(bitset);
    CacheShard& shard = *_cache[depth][shard_idx];

    std::lock_guard<std::mutex> lock(shard.mtx);
    auto it = shard.table.find(bitset);
    if (it != shard.table.end() && it->second.is_set()) {
        statistics::increment_cache();
        return true;
    }
    return false;
}

void Cache::store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree) {
    if (!use_caching || !tree->is_initialized()) return;

    const auto& bitset = data.get_bitset();
    if (!bitset.is_hash_set()) const_cast<DataviewBitset&>(bitset).set_hash(std::hash<DataviewBitset>()(bitset));

    int shard_idx = get_shard_idx(bitset);
    CacheShard& shard = *_cache[depth][shard_idx];

    std::lock_guard<std::mutex> lock(shard.mtx);
    shard.table[bitset] = CacheEntry(depth, tree);
}

std::shared_ptr<Tree> Cache::retrieve(const Dataview& data, int depth) {
    const auto& bitset = data.get_bitset();
    int shard_idx = get_shard_idx(bitset);
    CacheShard& shard = *_cache[depth][shard_idx];

    std::lock_guard<std::mutex> lock(shard.mtx);
    auto it = shard.table.find(bitset);
    if (it != shard.table.end()) return it->second.solution;
    
    return std::make_shared<Tree>();
}