#include "GPUBruteForceSolver.h"

#ifdef USE_CUDA

#include <algorithm>
#include <iostream>
#include <utility>

void check_leaf_regions_depth2(const Tree& t, const Dataview& dv) {
    if (!t.is_internal() || !t.left || !t.right) return;
    if (!t.left->is_internal() || !t.right->is_internal()) return;

    int f0 = t.split_feature;
    float thr0 = t.split_threshold;

    int fL = t.left->split_feature;
    float thrL = t.left->split_threshold;

    int fR = t.right->split_feature;
    float thrR = t.right->split_threshold;

    int labLL = t.left->left->label;
    int labLR = t.left->right->label;
    int labRL = t.right->left->label;
    int labRR = t.right->right->label;

    auto count_region = [&](auto pred) {
        int tot = 0;
        int c1 = 0;
        for (const auto& inst : dv.get_sorted_dataset_feature(0)) {
            int i = inst.data_point_index;
            if (!pred(i)) continue;
            tot++;
            int y = dv.get_unsorted_dataset_feature(0)[i].label;
            c1 += (y == 1);
        }
        return std::pair<int, int>(tot, c1);
    };

    auto rootL = [&](int i) { return dv.get_unsorted_dataset_feature(f0)[i].value <= thr0; };
    auto rootR = [&](int i) { return !rootL(i); };

    auto LL = [&](int i) { return rootL(i) && dv.get_unsorted_dataset_feature(fL)[i].value <= thrL; };
    auto LR = [&](int i) { return rootL(i) && dv.get_unsorted_dataset_feature(fL)[i].value > thrL; };
    auto RL = [&](int i) { return rootR(i) && dv.get_unsorted_dataset_feature(fR)[i].value <= thrR; };
    auto RR = [&](int i) { return rootR(i) && dv.get_unsorted_dataset_feature(fR)[i].value > thrR; };

    auto check = [&](const char* name, auto pred, int leaf_label) {
        auto [tot, c1] = count_region(pred);
        int c0 = tot - c1;
        int maj = (c1 > c0) ? 1 : 0;
        int err = tot - std::max(c0, c1);
        std::cout << "[LEAF] " << name
                  << " tot=" << tot << " c0=" << c0 << " c1=" << c1
                  << " maj=" << maj << " leaf=" << leaf_label
                  << " err=" << err
                  << (maj == leaf_label ? " OK" : " MISMATCH")
                  << "\n";
    };

    check("LL", LL, labLL);
    check("LR", LR, labLR);
    check("RL", RL, labRL);
    check("RR", RR, labRR);
}

#endif
