#ifndef GPU_BRUTE_FORCE_SOLVER_H
#define GPU_BRUTE_FORCE_SOLVER_H

#include <memory>
#include <vector>
#include "dataview.h"
#include "configuration.h"
#include "tree.h"

class GPUBruteForceSolver {
public:
    // 1. Veriyi GPU'ya yükler (Süre ölçümünden önce çağrılır)
    static void Initialize(const Dataview& full_dataset, const Configuration& config);

    // 2. GPU belleğini temizler (Program sonunda çağrılır)
    static void FreeMemory();

    // 3. Hesaplamayı yapar (Recursive olarak çağrılır, veri yüklemez)
    static void solve(
        const Dataview& dataview,
        const Configuration& config,
        std::shared_ptr<Tree>& out_tree
    );

    static void RunDeferredChecks();
};

#ifdef USE_CUDA
void check_leaf_regions_depth2(const Tree& t, const Dataview& dv);
#endif

#endif
