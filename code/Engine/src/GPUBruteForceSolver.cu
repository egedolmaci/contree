#include "GPUBruteForceSolver.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <omp.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do {                                      \
  cudaError_t err = (x);                                        \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
    abort();                                                    \
  }                                                            \
} while(0)
#endif

static inline long long popc_words(const std::vector<uint32_t>& v) {
    long long s = 0;
    for (uint32_t x : v) s += __builtin_popcount(x);
    return s;
}
static inline long long popc_words_ptr(const uint32_t* v, int words) {
    long long s = 0;
    for (int i = 0; i < words; ++i) s += __builtin_popcount(v[i]);
    return s;
}

typedef unsigned int uint32_t;
#define BITS_PER_WORD 32
#define MAX_CLASSES 16 

// --------------------------------------------------------
// GLOBAL MEMORY
// --------------------------------------------------------
static uint32_t* d_class_masks = nullptr; 
static uint32_t* d_candidate_masks = nullptr;
struct BestPack {
    int err;
    int idx;
    int bL;
    int bR;
    int clsLL, clsLR, clsRL, clsRR;
};

struct GPUThreadContext {
    cudaStream_t stream = nullptr;

    // per-thread device scratch
    uint32_t* d_active = nullptr;
    int* d_err = nullptr;
    int* d_bL  = nullptr;
    int* d_bR  = nullptr;
    int* d_leafs = nullptr;

    BestPack* d_block_best = nullptr;
    BestPack* d_best_one = nullptr;

    // per-thread pinned host scratch
    uint32_t* h_active_pinned = nullptr;
    size_t h_active_pinned_bytes = 0;
    BestPack* h_best_pinned = nullptr;

    int blocks_for_reduce = 0;
    int num_words = 0;
    int num_candidates = 0;
    bool allocated = false;
};

static thread_local GPUThreadContext tls;
static int global_num_words = 0;
static int global_num_classes = 0;
static int global_num_candidates = 0;
static int global_num_rows = 0;
static std::atomic<bool> is_initialized{false};
static std::mutex init_mutex;
static std::vector<uint32_t> h_class_masks_host;

struct HostCandidateInfo {
    int feature_index;
    float threshold;
};
static std::vector<HostCandidateInfo> h_candidates_info;

struct DeferredCheck {
    std::shared_ptr<Tree> tree;
    Dataview dataview;
};
static std::vector<DeferredCheck> g_deferred_checks;

static void queue_deferred_check(const std::shared_ptr<Tree>& tree, const Dataview& dataview) {
    #pragma omp critical(gpu_check_queue)
    {
        g_deferred_checks.push_back({tree, dataview});
    }
}

static inline int predict_row(const Tree& t, const Dataview& dv, int row_idx) {
    const Tree* cur = &t;
    while (cur->is_internal()) {
        int f = cur->get_split_feature();
        float thr = cur->get_split_threshold();
        float x = dv.get_unsorted_dataset_feature(f)[row_idx].value;
        cur = (x <= thr) ? cur->left.get() : cur->right.get();
    }
    return cur->get_label();
}

static inline int cpu_mis_on_dataview(const Tree& t, const Dataview& dv) {
    const auto& rows = dv.get_sorted_dataset_feature(0);
    int mis = 0;
    for (const auto& inst : rows) {
        int i = inst.data_point_index;
        int y = dv.get_unsorted_dataset_feature(0)[i].label;
        int yhat = predict_row(t, dv, i);
        mis += (yhat != y);
    }
    return mis;
}

static void free_thread_context(GPUThreadContext& ctx) {
    if (ctx.stream) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    if (ctx.d_best_one) cudaFree(ctx.d_best_one);
    if (ctx.d_block_best) cudaFree(ctx.d_block_best);
    if (ctx.h_best_pinned) cudaFreeHost(ctx.h_best_pinned);
    if (ctx.d_leafs) cudaFree(ctx.d_leafs);
    if (ctx.d_bR) cudaFree(ctx.d_bR);
    if (ctx.d_bL) cudaFree(ctx.d_bL);
    if (ctx.d_err) cudaFree(ctx.d_err);
    if (ctx.d_active) cudaFree(ctx.d_active);
    if (ctx.h_active_pinned) cudaFreeHost(ctx.h_active_pinned);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);

    ctx = GPUThreadContext{};
}

static void EnsureThreadContextAllocated() {
    if (tls.allocated) {
        if (tls.num_words == global_num_words && tls.num_candidates == global_num_candidates) {
            return;
        }
        free_thread_context(tls);
    }
    if (!is_initialized.load()) {
        std::fprintf(stderr, "[GPU ERROR] Initialize must be called before solve().\n");
        std::abort();
    }

    CUDA_CHECK(cudaStreamCreate(&tls.stream));
    CUDA_CHECK(cudaMalloc(&tls.d_active, global_num_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tls.d_err, global_num_candidates * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tls.d_bL, global_num_candidates * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tls.d_bR, global_num_candidates * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tls.d_leafs, global_num_candidates * 4 * sizeof(int)));

    int threads = 256;
    tls.blocks_for_reduce = (global_num_candidates + threads - 1) / threads;
    if (global_num_candidates > 0) {
        CUDA_CHECK(cudaMalloc(&tls.d_block_best, tls.blocks_for_reduce * sizeof(BestPack)));
        CUDA_CHECK(cudaMalloc(&tls.d_best_one, sizeof(BestPack)));
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&tls.h_best_pinned),
            sizeof(BestPack),
            cudaHostAllocDefault
        ));
    }

    tls.h_active_pinned_bytes = static_cast<size_t>(global_num_words) * sizeof(uint32_t);
    CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&tls.h_active_pinned),
        tls.h_active_pinned_bytes,
        cudaHostAllocDefault
    ));

    tls.num_words = global_num_words;
    tls.num_candidates = global_num_candidates;
    tls.allocated = true;
}

__device__ __forceinline__ bool better(const BestPack& a, const BestPack& b) {
    if (a.err != b.err) return a.err < b.err;
    return a.idx < b.idx;
}

__global__ void reduce_best_stage1(
    const int* __restrict__ d_err,
    const int* __restrict__ d_bL,
    const int* __restrict__ d_bR,
    const int* __restrict__ d_leafs,
    int n,
    BestPack* __restrict__ out_block_best
) {
    extern __shared__ BestPack sh[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    BestPack cur;
    if (i < n) {
        cur.err  = d_err[i];
        cur.idx  = i;
        cur.bL   = d_bL[i];
        cur.bR   = d_bR[i];
        cur.clsLL = d_leafs[i * 4 + 0];
        cur.clsLR = d_leafs[i * 4 + 1];
        cur.clsRL = d_leafs[i * 4 + 2];
        cur.clsRR = d_leafs[i * 4 + 3];
    } else {
        cur.err = INT_MAX;
        cur.idx = INT_MAX;
        cur.bL = cur.bR = -1;
        cur.clsLL = cur.clsLR = cur.clsRL = cur.clsRR = -1;
    }

    sh[tid] = cur;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            BestPack other = sh[tid + offset];
            if (better(other, sh[tid])) sh[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) out_block_best[blockIdx.x] = sh[0];
}

__global__ void reduce_best_stage2(
    const BestPack* __restrict__ in,
    int n,
    BestPack* __restrict__ out_one
) {
    extern __shared__ BestPack sh[];
    int tid = threadIdx.x;

    BestPack cur;
    cur.err = INT_MAX;
    cur.idx = INT_MAX;
    cur.bL = cur.bR = -1;
    cur.clsLL = cur.clsLR = cur.clsRL = cur.clsRR = -1;

    for (int i = tid; i < n; i += blockDim.x) {
        BestPack x = in[i];
        if (better(x, cur)) cur = x;
    }

    sh[tid] = cur;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            BestPack other = sh[tid + offset];
            if (better(other, sh[tid])) sh[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) *out_one = sh[0];
}

// --------------------------------------------------------
// KERNEL: Bitset Based Depth-2 Solver (With Leaf Class Tracking)
// --------------------------------------------------------
__device__ __forceinline__ void warpReduceMinWithIndex(int& val, int& idx, int& cls1, int& cls2) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        int other_cls1 = __shfl_down_sync(0xFFFFFFFF, cls1, offset);
        int other_cls2 = __shfl_down_sync(0xFFFFFFFF, cls2, offset);
        if (other_val < val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
            cls1 = other_cls1;
            cls2 = other_cls2;
        }
    }
}

__device__ __forceinline__ void warpReduceMinWithIndexK2(
    int& val, int& idx, int& cls1, int& cls2
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        int other_cls1 = __shfl_down_sync(0xFFFFFFFF, cls1, offset);
        int other_cls2 = __shfl_down_sync(0xFFFFFFFF, cls2, offset);
        if (other_val < val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
            cls1 = other_cls1;
            cls2 = other_cls2;
        }
    }
}

__global__ void solve_depth2_binary_kernel(
    const uint32_t* __restrict__ candidate_masks,
    const uint32_t* __restrict__ pos_mask,
    const uint32_t* __restrict__ active_row_mask,
    int num_candidates,
    int num_words,
    int* __restrict__ out_errors,
    int* __restrict__ out_best_L_idx,
    int* __restrict__ out_best_R_idx,
    int* __restrict__ out_leaf_classes
) {
    if (blockDim.x > 256) return;
    int root_idx = blockIdx.x;
    if (root_idx >= num_candidates) return;

    const uint32_t* root_ptr = &candidate_masks[root_idx * num_words];

    int totL_local = 0, posL_local = 0;
    int totR_local = 0, posR_local = 0;

    for (int w = threadIdx.x; w < num_words; w += blockDim.x) {
        uint32_t v = active_row_mask[w];
        if (!v) continue;

        uint32_t r = root_ptr[w];
        uint32_t L = r & v;
        uint32_t R = (~r) & v;
        uint32_t p = pos_mask[w];

        totL_local += __popc(L);
        posL_local += __popc(L & p);
        totR_local += __popc(R);
        posR_local += __popc(R & p);
    }

    __shared__ int s_totL, s_posL, s_totR, s_posR;
    __shared__ int s_w_totL[8], s_w_posL[8], s_w_totR[8], s_w_posR[8];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        totL_local += __shfl_down_sync(0xFFFFFFFF, totL_local, offset);
        posL_local += __shfl_down_sync(0xFFFFFFFF, posL_local, offset);
        totR_local += __shfl_down_sync(0xFFFFFFFF, totR_local, offset);
        posR_local += __shfl_down_sync(0xFFFFFFFF, posR_local, offset);
    }
    if (lane == 0) {
        s_w_totL[warp] = totL_local;
        s_w_posL[warp] = posL_local;
        s_w_totR[warp] = totR_local;
        s_w_posR[warp] = posR_local;
    }
    __syncthreads();

    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        int tL = (lane < nw) ? s_w_totL[lane] : 0;
        int pL = (lane < nw) ? s_w_posL[lane] : 0;
        int tR = (lane < nw) ? s_w_totR[lane] : 0;
        int pR = (lane < nw) ? s_w_posR[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            tL += __shfl_down_sync(0xFFFFFFFF, tL, offset);
            pL += __shfl_down_sync(0xFFFFFFFF, pL, offset);
            tR += __shfl_down_sync(0xFFFFFFFF, tR, offset);
            pR += __shfl_down_sync(0xFFFFFFFF, pR, offset);
        }
        if (lane == 0) {
            s_totL = tL; s_posL = pL;
            s_totR = tR; s_posR = pR;
        }
    }
    __syncthreads();

    const int TOTL = s_totL, POSL = s_posL;
    const int TOTR = s_totR, POSR = s_posR;

    int my_best_L = INT_MAX, my_idx_L = -1, my_cls_LL = 0, my_cls_LR = 0;
    int my_best_R = INT_MAX, my_idx_R = -1, my_cls_RL = 0, my_cls_RR = 0;

    for (int c_idx = threadIdx.x; c_idx < num_candidates; c_idx += blockDim.x) {
        const uint32_t* child_ptr = &candidate_masks[c_idx * num_words];

        int totLL = 0, posLL = 0;
        int totRL = 0, posRL = 0;

        for (int w = 0; w < num_words; ++w) {
            uint32_t v = active_row_mask[w];
            if (!v) continue;

            uint32_t r = root_ptr[w];
            uint32_t c = child_ptr[w];
            uint32_t p = pos_mask[w];

            uint32_t rnv = (~r) & v;
            uint32_t rv = r & v;
            uint32_t cv = c & v;

            uint32_t mLL = rv & cv;
            uint32_t mRL = rnv & cv;

            totLL += __popc(mLL);
            posLL += __popc(mLL & p);
            totRL += __popc(mRL);
            posRL += __popc(mRL & p);
        }

        int totLR = TOTL - totLL;
        int posLR = POSL - posLL;
        int totRR = TOTR - totRL;
        int posRR = POSR - posRL;

        int errLL = (posLL < (totLL - posLL)) ? posLL : (totLL - posLL);
        int errLR = (posLR < (totLR - posLR)) ? posLR : (totLR - posLR);
        int errRL = (posRL < (totRL - posRL)) ? posRL : (totRL - posRL);
        int errRR = (posRR < (totRR - posRR)) ? posRR : (totRR - posRR);

        int errL = errLL + errLR;
        int errR = errRL + errRR;

        int clsLL = (posLL > (totLL - posLL)) ? 1 : 0;
        int clsLR = (posLR > (totLR - posLR)) ? 1 : 0;
        int clsRL = (posRL > (totRL - posRL)) ? 1 : 0;
        int clsRR = (posRR > (totRR - posRR)) ? 1 : 0;

        if (errL < my_best_L || (errL == my_best_L && c_idx < my_idx_L)) {
            my_best_L = errL; my_idx_L = c_idx;
            my_cls_LL = clsLL; my_cls_LR = clsLR;
        }
        if (errR < my_best_R || (errR == my_best_R && c_idx < my_idx_R)) {
            my_best_R = errR; my_idx_R = c_idx;
            my_cls_RL = clsRL; my_cls_RR = clsRR;
        }
    }

    __shared__ int s_w_best_L[8], s_w_idx_L[8], s_w_cls_LL[8], s_w_cls_LR[8];
    __shared__ int s_w_best_R[8], s_w_idx_R[8], s_w_cls_RL[8], s_w_cls_RR[8];
    __shared__ int s_best_L, s_idx_L, s_cls_LL, s_cls_LR;
    __shared__ int s_best_R, s_idx_R, s_cls_RL, s_cls_RR;

    warpReduceMinWithIndexK2(my_best_L, my_idx_L, my_cls_LL, my_cls_LR);
    warpReduceMinWithIndexK2(my_best_R, my_idx_R, my_cls_RL, my_cls_RR);

    if (lane == 0) {
        s_w_best_L[warp] = my_best_L;
        s_w_idx_L[warp] = my_idx_L;
        s_w_cls_LL[warp] = my_cls_LL;
        s_w_cls_LR[warp] = my_cls_LR;

        s_w_best_R[warp] = my_best_R;
        s_w_idx_R[warp] = my_idx_R;
        s_w_cls_RL[warp] = my_cls_RL;
        s_w_cls_RR[warp] = my_cls_RR;
    }
    __syncthreads();

    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;

        int vL = (lane < nw) ? s_w_best_L[lane] : INT_MAX;
        int iL = (lane < nw) ? s_w_idx_L[lane] : -1;
        int cLL = (lane < nw) ? s_w_cls_LL[lane] : 0;
        int cLR = (lane < nw) ? s_w_cls_LR[lane] : 0;

        int vR = (lane < nw) ? s_w_best_R[lane] : INT_MAX;
        int iR = (lane < nw) ? s_w_idx_R[lane] : -1;
        int cRL = (lane < nw) ? s_w_cls_RL[lane] : 0;
        int cRR = (lane < nw) ? s_w_cls_RR[lane] : 0;

        warpReduceMinWithIndexK2(vL, iL, cLL, cLR);
        warpReduceMinWithIndexK2(vR, iR, cRL, cRR);

        if (lane == 0) {
            s_best_L = vL; s_idx_L = iL; s_cls_LL = cLL; s_cls_LR = cLR;
            s_best_R = vR; s_idx_R = iR; s_cls_RL = cRL; s_cls_RR = cRR;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        out_errors[root_idx] = s_best_L + s_best_R;
        out_best_L_idx[root_idx] = s_idx_L;
        out_best_R_idx[root_idx] = s_idx_R;
        out_leaf_classes[root_idx * 4 + 0] = s_cls_LL;
        out_leaf_classes[root_idx * 4 + 1] = s_cls_LR;
        out_leaf_classes[root_idx * 4 + 2] = s_cls_RL;
        out_leaf_classes[root_idx * 4 + 3] = s_cls_RR;
    }
}

__global__ void solve_depth2_fast_kernel(
    const uint32_t* __restrict__ candidate_masks, 
    const uint32_t* __restrict__ class_masks,     
    const uint32_t* __restrict__ active_row_mask, 
    int num_candidates,
    int num_words,
    int num_classes,
    int* out_errors,
    int* out_best_L_idx,
    int* out_best_R_idx,
    int* out_leaf_classes // [4 * num_candidates] (LL, LR, RL, RR)
) {
    if (blockDim.x > 256) return;
    int root_idx = blockIdx.x;
    if (root_idx >= num_candidates) return;

    const uint32_t* root_ptr = &candidate_masks[root_idx * num_words];

    // Thread Local Variables (Register Optimization)
    int my_best_L = 2147483647; int my_idx_L = -1; int my_LL = -1; int my_LR = -1;
    int my_best_R = 2147483647; int my_idx_R = -1; int my_RL = -1; int my_RR = -1;

    for (int c_idx = threadIdx.x; c_idx < num_candidates; c_idx += blockDim.x) {
        const uint32_t* child_ptr = &candidate_masks[c_idx * num_words];

        int cnt_LL[MAX_CLASSES] = {0};
        int cnt_LR[MAX_CLASSES] = {0};
        int cnt_RL[MAX_CLASSES] = {0};
        int cnt_RR[MAX_CLASSES] = {0};

        for (int w = 0; w < num_words; ++w) {
            uint32_t v = active_row_mask[w];
            if (v == 0) continue; 

            uint32_t r = root_ptr[w];
            uint32_t c = child_ptr[w];

            uint32_t m_LL = r & c & v;
            uint32_t m_LR = r & (~c) & v;
            uint32_t m_RL = (~r) & c & v;
            uint32_t m_RR = (~r) & (~c) & v;

            for (int k = 0; k < num_classes; ++k) {
                uint32_t cls = class_masks[k * num_words + w];
                if (m_LL) cnt_LL[k] += __popc(m_LL & cls);
                if (m_LR) cnt_LR[k] += __popc(m_LR & cls);
                if (m_RL) cnt_RL[k] += __popc(m_RL & cls);
                if (m_RR) cnt_RR[k] += __popc(m_RR & cls);
            }
        }

        // Helper to find Majority Class and Error
        int win_LL = -1, win_LR = -1, win_RL = -1, win_RR = -1;
        auto calc = [&](int* arr, int& win) {
            int tot = 0, mx = -1;
            for(int i=0; i<num_classes; ++i) {
                tot += arr[i];
                if (arr[i] > mx) { mx = arr[i]; win = i; }
            }
            return tot - mx;
        };

        int err_L = calc(cnt_LL, win_LL) + calc(cnt_LR, win_LR);
        int err_R = calc(cnt_RL, win_RL) + calc(cnt_RR, win_RR);

        if (err_L < my_best_L || (err_L == my_best_L && c_idx < my_idx_L)) {
            my_best_L = err_L; my_idx_L = c_idx;
            my_LL = win_LL; my_LR = win_LR;
        }
        if (err_R < my_best_R || (err_R == my_best_R && c_idx < my_idx_R)) {
            my_best_R = err_R; my_idx_R = c_idx;
            my_RL = win_RL; my_RR = win_RR;
        }
    }

    // Block-level reduction without races (warp then block)
    __shared__ int s_warp_best_L[8], s_warp_idx_L[8], s_warp_cls_LL[8], s_warp_cls_LR[8];
    __shared__ int s_warp_best_R[8], s_warp_idx_R[8], s_warp_cls_RL[8], s_warp_cls_RR[8];
    __shared__ int s_best_err_L, s_idx_L, s_cls_LL, s_cls_LR;
    __shared__ int s_best_err_R, s_idx_R, s_cls_RL, s_cls_RR;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    warpReduceMinWithIndex(my_best_L, my_idx_L, my_LL, my_LR);
    warpReduceMinWithIndex(my_best_R, my_idx_R, my_RL, my_RR);

    if (lane == 0) {
        s_warp_best_L[warp_id] = my_best_L;
        s_warp_idx_L[warp_id] = my_idx_L;
        s_warp_cls_LL[warp_id] = my_LL;
        s_warp_cls_LR[warp_id] = my_LR;

        s_warp_best_R[warp_id] = my_best_R;
        s_warp_idx_R[warp_id] = my_idx_R;
        s_warp_cls_RL[warp_id] = my_RL;
        s_warp_cls_RR[warp_id] = my_RR;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        int vL = (lane < num_warps) ? s_warp_best_L[lane] : 2147483647;
        int iL = (lane < num_warps) ? s_warp_idx_L[lane] : -1;
        int cLL = (lane < num_warps) ? s_warp_cls_LL[lane] : -1;
        int cLR = (lane < num_warps) ? s_warp_cls_LR[lane] : -1;

        int vR = (lane < num_warps) ? s_warp_best_R[lane] : 2147483647;
        int iR = (lane < num_warps) ? s_warp_idx_R[lane] : -1;
        int cRL = (lane < num_warps) ? s_warp_cls_RL[lane] : -1;
        int cRR = (lane < num_warps) ? s_warp_cls_RR[lane] : -1;

        warpReduceMinWithIndex(vL, iL, cLL, cLR);
        warpReduceMinWithIndex(vR, iR, cRL, cRR);

        if (lane == 0) {
            s_best_err_L = vL; s_idx_L = iL; s_cls_LL = cLL; s_cls_LR = cLR;
            s_best_err_R = vR; s_idx_R = iR; s_cls_RL = cRL; s_cls_RR = cRR;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        out_errors[root_idx] = s_best_err_L + s_best_err_R;
        out_best_L_idx[root_idx] = s_idx_L;
        out_best_R_idx[root_idx] = s_idx_R;
        
        // Save Leaf Classes
        out_leaf_classes[root_idx * 4 + 0] = s_cls_LL;
        out_leaf_classes[root_idx * 4 + 1] = s_cls_LR;
        out_leaf_classes[root_idx * 4 + 2] = s_cls_RL;
        out_leaf_classes[root_idx * 4 + 3] = s_cls_RR;
    }
}

// --------------------------------------------------------
// HOST: Initialize
// --------------------------------------------------------
void GPUBruteForceSolver::Initialize(const Dataview& full_dataset, const Configuration& config) {
    // Initialization must happen before solve(); guarded for concurrent calls.
    std::lock_guard<std::mutex> lock(init_mutex);
    if (is_initialized.load()) return;

    int N = full_dataset.get_dataset_size();
    int F = full_dataset.get_feature_number();
    global_num_classes = full_dataset.get_class_number();
    global_num_words = (N + BITS_PER_WORD - 1) / BITS_PER_WORD;
    global_num_rows = N;

    // Hard Limit Check
    if (global_num_classes > MAX_CLASSES) {
        std::cerr << "[GPU ERROR] Too many classes (" << global_num_classes 
                  << "). Increase MAX_CLASSES." << std::endl;
        exit(1);
    }


    // 1. Class Masks
    std::vector<uint32_t> h_class_masks(global_num_classes * global_num_words, 0);
    const auto& label_feat = full_dataset.get_unsorted_dataset_feature(0); 

    for (int i = 0; i < N; ++i) {
        int label = label_feat[i].label;
        if (label < 0 || label >= global_num_classes) {
            std::fprintf(stderr, "[GPU ERROR] Bad label=%d at i=%d\n", label, i);
            std::abort();
        }
        h_class_masks[label * global_num_words + (i / 32)] |= (1U << (i % 32));
    }

    if (config.print_logs && N > 0) {
        const auto& col0 = full_dataset.get_unsorted_dataset_feature(0);
        float mn = col0[0].value;
        float mx = col0[0].value;
        bool all_int = true;
        for (int i = 0; i < N; ++i) {
            float v = col0[i].value;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            if (std::fabs(v - std::round(v)) > 1e-6f) all_int = false;
        }
        bool looks_like_label = all_int && mn >= 0.0f && mx <= static_cast<float>(global_num_classes - 1);
        std::cout << "[GPU] F=" << F << " classes=" << global_num_classes << "\n";
        std::cout << "[GPU] feat0 value range: " << mn << " .. " << mx << "\n";
        std::cout << "[GPU] feat0 looks like label: " << (looks_like_label ? "yes" : "no") << "\n";
    }

    // 2. Candidates
    std::vector<uint32_t> h_cand_masks;
    h_candidates_info.clear();
    std::vector<int> unique_counts;
    if (config.print_logs) unique_counts.reserve(F);

    for (int f = 0; f < F; ++f) {
        struct Pair { float val; int orig_idx; };
        std::vector<Pair> data(N);
        const auto& col = full_dataset.get_unsorted_dataset_feature(f);
        
        for(int i=0; i<N; ++i) data[i] = {col[i].value, i};

        std::sort(data.begin(), data.end(), [](const Pair& a, const Pair& b){
            return a.val < b.val;
        });

        std::vector<float> unique_vals;
        unique_vals.reserve(N);
        for (int i = 0; i < N; ++i) {
            float v = data[i].val;
            if (unique_vals.empty() || (v - unique_vals.back()) >= EPSILON) {
                unique_vals.push_back(v);
            }
        }
        if (config.print_logs) unique_counts.push_back(static_cast<int>(unique_vals.size()));

        // Use midpoints as thresholds to match CPU behavior (and reduce count by 1).
        // Cap thresholds per feature to keep GPU depth-2 brute force tractable.
        std::vector<float> selected_thresholds;
        if (unique_vals.size() > 1) {
            int usable = static_cast<int>(unique_vals.size() - 1);
            int take = std::min(std::max(config.max_thresholds_per_feature, 1), usable);
            selected_thresholds.reserve(take);
            for (int t = 1; t <= take; ++t) {
                int i = static_cast<int>((static_cast<int64_t>(t) * usable) / (take + 1));
                float mid = (unique_vals[i] + unique_vals[i + 1]) / 2.0f;
                selected_thresholds.push_back(mid);
            }
        }
        std::sort(selected_thresholds.begin(), selected_thresholds.end());
        selected_thresholds.erase(
            std::unique(selected_thresholds.begin(), selected_thresholds.end()),
            selected_thresholds.end()
        );
        std::vector<uint32_t> current_mask(global_num_words, 0);
        int data_ptr = 0;

        for (float thr : selected_thresholds) {
            while(data_ptr < N && data[data_ptr].val <= thr) {
                int oid = data[data_ptr].orig_idx;
                current_mask[oid / 32] |= (1U << (oid % 32));
                data_ptr++;
            }
            h_candidates_info.push_back({f, thr});
            h_cand_masks.insert(h_cand_masks.end(), current_mask.begin(), current_mask.end());
        }
    }

    global_num_candidates = h_candidates_info.size();
    if (config.print_logs) {
        for (int f = 0; f < F; ++f) {
            std::cout << "[GPU] unique_vals feature " << f
                      << ": " << unique_counts[f] << std::endl;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_class_masks, h_class_masks.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_masks, h_cand_masks.size() * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(
        d_class_masks,
        h_class_masks.data(),
        h_class_masks.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_candidate_masks,
        h_cand_masks.data(),
        h_cand_masks.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));
    h_class_masks_host = h_class_masks;

    is_initialized.store(true);
    if (config.print_logs) {
        std::cout << "[GPU] thread-safe mode enabled (per-thread streams/buffers)\n";
    }
}

void GPUBruteForceSolver::FreeMemory() {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (d_candidate_masks) cudaFree(d_candidate_masks);
    if (d_class_masks) cudaFree(d_class_masks);
    d_candidate_masks = nullptr;
    d_class_masks = nullptr;
    h_class_masks_host.clear();
    is_initialized.store(false);
}

// --------------------------------------------------------
// HOST: Solve
// --------------------------------------------------------
void GPUBruteForceSolver::solve(
    const Dataview& dataview,
    const Configuration& config,
    std::shared_ptr<Tree>& out_tree
) {
    if (!out_tree) out_tree = std::make_shared<Tree>();
    if (!is_initialized.load()) {
        std::fprintf(
            stderr,
            "[GPU ERROR] GPUBruteForceSolver::Initialize must be called with the full dataset before solve().\n"
        );
        std::abort();
    }
    EnsureThreadContextAllocated();
    auto& ctx = tls;

    int N = dataview.get_dataset_size();
    int unsorted_size = static_cast<int>(dataview.get_unsorted_dataset_feature(0).size());
    if (unsorted_size != global_num_rows) {
        std::fprintf(
            stderr,
            "[GPU ERROR] Unsorted feature size=%d but global=%d\n",
            unsorted_size, global_num_rows
        );
        std::abort();
    }

    if (!ctx.h_active_pinned) {
        std::fprintf(stderr, "[GPU ERROR] Pinned host buffer not initialized.\n");
        std::abort();
    }
    if (ctx.h_active_pinned_bytes != static_cast<size_t>(global_num_words) * sizeof(uint32_t)) {
        std::fprintf(stderr, "[GPU ERROR] Pinned buffer size mismatch.\n");
        std::abort();
    }
    std::memset(ctx.h_active_pinned, 0, ctx.h_active_pinned_bytes);
    uint32_t* h_active = ctx.h_active_pinned;
    const auto& instances = dataview.get_sorted_dataset_feature(0);
    int max_idx = -1;
    for (const auto& inst : instances) {
        int idx = inst.data_point_index;
        if (idx < 0 || idx >= global_num_rows) {
            std::printf("[GPU] ERROR invalid global index=%d\n", idx);
            abort();
        }
        if (idx > max_idx) max_idx = idx;
        h_active[idx / 32] |= (1U << (idx % 32));
    }

    int pad_bits = global_num_words * 32 - global_num_rows;
    if (pad_bits < 0 || pad_bits >= 32) {
        std::printf(
            "[GPU] ERROR pad=%d global_N=%d global_words=%d\n",
            pad_bits, global_num_rows, global_num_words
        );
        abort();
    }
    if (pad_bits > 0) {
        uint32_t keep = 0xFFFFFFFFu >> pad_bits;
        h_active[global_num_words - 1] &= keep;
    }
    
    static const bool kZeroMaskTrap = false;
    static int did_zero_trap = 0;
    if (kZeroMaskTrap && !did_zero_trap) {
        did_zero_trap = 1;
        std::memset(h_active, 0, ctx.h_active_pinned_bytes);
        if (pad_bits > 0) {
            uint32_t keep = 0xFFFFFFFFu >> pad_bits;
            h_active[global_num_words - 1] &= keep;
        }
    }

    int active_popcount = static_cast<int>(popc_words_ptr(h_active, global_num_words));
    if (active_popcount == 0) {
        int best_label = 0;
        int best_count = -1;
        const auto& freq = dataview.get_label_frequency();
        for (int label = 0; label < static_cast<int>(freq.size()); ++label) {
            if (freq[label] > best_count) {
                best_count = freq[label];
                best_label = label;
            }
        }
        int mis = dataview.get_dataset_size() - best_count;
        out_tree->make_leaf(best_label, mis);
        return;
    }

    size_t active_bytes = static_cast<size_t>(global_num_words) * sizeof(uint32_t);
    CUDA_CHECK(cudaMemcpyAsync(
        ctx.d_active,
        ctx.h_active_pinned,
        active_bytes,
        cudaMemcpyHostToDevice,
        ctx.stream
    ));

    solve_depth2_fast_kernel<<<global_num_candidates, 256, 0, ctx.stream>>>(
        d_candidate_masks,
        d_class_masks,
        ctx.d_active,
        global_num_candidates,
        global_num_words,
        global_num_classes,
        ctx.d_err, ctx.d_bL, ctx.d_bR, ctx.d_leafs
    );
    CUDA_CHECK(cudaGetLastError());

    int best_val = INT_MAX;
    int best_idx = -1;
    if (global_num_candidates > 0) {
        int threads = 256;
        int blocks = ctx.blocks_for_reduce;

        reduce_best_stage1<<<blocks, threads, threads * sizeof(BestPack), ctx.stream>>>(
            ctx.d_err,
            ctx.d_bL,
            ctx.d_bR,
            ctx.d_leafs,
            global_num_candidates,
            ctx.d_block_best
        );
        CUDA_CHECK(cudaGetLastError());

        int threads2 = 1024;
        if (blocks < threads2) {
            threads2 = 1;
            while (threads2 < blocks) threads2 <<= 1;
        }

        reduce_best_stage2<<<1, threads2, threads2 * sizeof(BestPack), ctx.stream>>>(
            ctx.d_block_best,
            blocks,
            ctx.d_best_one
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            ctx.h_best_pinned,
            ctx.d_best_one,
            sizeof(BestPack),
            cudaMemcpyDeviceToHost,
            ctx.stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        BestPack best = *ctx.h_best_pinned;
        best_idx = (best.err == INT_MAX) ? -1 : best.idx;
        best_val = best.err;
            if (best_idx != -1) {
            if (config.print_logs) {
                std::cout << "[GPU] root split feature="
                          << h_candidates_info[best_idx].feature_index
                          << " threshold="
                          << h_candidates_info[best_idx].threshold
                          << std::endl;
            }
            int idx_L = best.bL;
            int idx_R = best.bR;
            int cls_LL = best.clsLL;
            int cls_LR = best.clsLR;
            int cls_RL = best.clsRL;
            int cls_RR = best.clsRR;

            std::vector<uint32_t> h_root_mask(global_num_words);
            CUDA_CHECK(cudaMemcpyAsync(
                h_root_mask.data(),
                d_candidate_masks + (best_idx * global_num_words),
                global_num_words * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                ctx.stream
            ));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

            if (config.print_logs) {
                int cpuL = 0, cpuR = 0;
                int f = h_candidates_info[best_idx].feature_index;
                float thr = h_candidates_info[best_idx].threshold;
                for (const auto& inst : dataview.get_sorted_dataset_feature(0)) {
                    int i = inst.data_point_index;
                    float x = dataview.get_unsorted_dataset_feature(f)[i].value;
                    if (x <= thr) cpuL++;
                    else cpuR++;
                }
                int gpuL = 0, gpuR = 0;
                for (int w = 0; w < global_num_words; ++w) {
                    uint32_t v = h_active[w];
                    uint32_t r = h_root_mask[w];
                    gpuL += __builtin_popcount(r & v);
                    gpuR += __builtin_popcount((~r) & v);
                }
                std::cout << "[CHECK] root split counts cpu(L,R)=("
                          << cpuL << "," << cpuR << ") gpu(L,R)=("
                          << gpuL << "," << gpuR << ")\n";
            }

            int count_L = 0;
            int count_R = 0;
            for (int w = 0; w < global_num_words; ++w) {
                uint32_t v = h_active[w];
                uint32_t r = h_root_mask[w];
                count_L += __builtin_popcount(r & v);
                count_R += __builtin_popcount((~r) & v);
            }
            std::vector<uint32_t> child_left(global_num_words);
            std::vector<uint32_t> child_right(global_num_words);
            for (int w = 0; w < global_num_words; ++w) {
                uint32_t v = h_active[w];
                uint32_t r = h_root_mask[w];
                child_left[w] = v & r;
                child_right[w] = v & (~r);
            }
            long long child_left_popc = popc_words_ptr(child_left.data(), global_num_words);
            long long child_right_popc = popc_words_ptr(child_right.data(), global_num_words);
            if (child_left_popc == 0 || child_right_popc == 0) {
                int best_label = 0;
                int best_count = -1;
                const auto& freq = dataview.get_label_frequency();
                for (int label = 0; label < static_cast<int>(freq.size()); ++label) {
                    if (freq[label] > best_count) {
                        best_count = freq[label];
                        best_label = label;
                    }
                }
                int mis = dataview.get_dataset_size() - best_count;
                out_tree->make_leaf(best_label, mis);
                return;
            }

            auto majority_from_mask = [&](const std::vector<uint32_t>& mask) {
                int best_label = 0;
                int best_count = -1;
                int total = 0;
                for (int k = 0; k < global_num_classes; ++k) {
                    int cnt = 0;
                    const uint32_t* cls = &h_class_masks_host[k * global_num_words];
                    for (int w = 0; w < global_num_words; ++w) {
                        cnt += __builtin_popcount(mask[w] & cls[w]);
                    }
                    total += cnt;
                    if (cnt > best_count) {
                        best_count = cnt;
                        best_label = k;
                    }
                }
                int mis = total - best_count;
                return std::pair<int, int>(best_label, mis);
            };

            auto build_leaf_masks = [&](int idx, const std::vector<uint32_t>& side_mask,
                                       std::vector<uint32_t>& mL,
                                       std::vector<uint32_t>& mR) {
                std::vector<uint32_t> split_mask(global_num_words);
                CUDA_CHECK(cudaMemcpyAsync(
                    split_mask.data(),
                    d_candidate_masks + (idx * global_num_words),
                    global_num_words * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    ctx.stream
                ));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                for (int w = 0; w < global_num_words; ++w) {
                    uint32_t v = side_mask[w];
                    uint32_t s = split_mask[w];
                    mL[w] = v & s;
                    mR[w] = v & (~s);
                }
            };

            auto L_node = std::make_shared<Tree>();
            if (idx_L != -1) {
                std::vector<uint32_t> mask_LL(global_num_words);
                std::vector<uint32_t> mask_LR(global_num_words);
                build_leaf_masks(idx_L, child_left, mask_LL, mask_LR);
                auto [label_LL, mis_LL] = majority_from_mask(mask_LL);
                auto [label_LR, mis_LR] = majority_from_mask(mask_LR);
                auto leaf_LL = std::make_shared<Tree>(); leaf_LL->make_leaf(label_LL, mis_LL);
                auto leaf_LR = std::make_shared<Tree>(); leaf_LR->make_leaf(label_LR, mis_LR);

                L_node->update_split(h_candidates_info[idx_L].feature_index,
                                     h_candidates_info[idx_L].threshold,
                                     leaf_LL, leaf_LR);
            } else {
                auto [best_label, mis] = majority_from_mask(child_left);
                L_node->make_leaf(best_label, mis);
            }

            auto R_node = std::make_shared<Tree>();
            if (idx_R != -1) {
                std::vector<uint32_t> mask_RL(global_num_words);
                std::vector<uint32_t> mask_RR(global_num_words);
                build_leaf_masks(idx_R, child_right, mask_RL, mask_RR);
                auto [label_RL, mis_RL] = majority_from_mask(mask_RL);
                auto [label_RR, mis_RR] = majority_from_mask(mask_RR);
                auto leaf_RL = std::make_shared<Tree>(); leaf_RL->make_leaf(label_RL, mis_RL);
                auto leaf_RR = std::make_shared<Tree>(); leaf_RR->make_leaf(label_RR, mis_RR);

                R_node->update_split(h_candidates_info[idx_R].feature_index,
                                     h_candidates_info[idx_R].threshold,
                                     leaf_RL, leaf_RR);
            } else {
                auto [best_label, mis] = majority_from_mask(child_right);
                R_node->make_leaf(best_label, mis);
            }

            out_tree->update_split(
                h_candidates_info[best_idx].feature_index,
                h_candidates_info[best_idx].threshold,
                L_node, R_node
            );

            if (config.print_logs) {
                if (config.defer_gpu_checks && omp_in_parallel()) {
                    queue_deferred_check(out_tree, dataview);
                } else {
                    check_leaf_regions_depth2(*out_tree, dataview);
                    int cpu_mis = cpu_mis_on_dataview(*out_tree, dataview);
                    std::cout << "[CHECK] GPU_score=" << out_tree->misclassification_score
                              << " CPU_eval=" << cpu_mis << "\n";
                    RUNTIME_ASSERT(out_tree->misclassification_score == cpu_mis,
                                   "GPU misclassification_score != CPU evaluated mis");
                }
            }
        }
    }

    if (best_idx == -1) {
        return;
    }
}

void GPUBruteForceSolver::RunDeferredChecks() {
    std::vector<DeferredCheck> pending;
    #pragma omp critical(gpu_check_queue)
    {
        pending.swap(g_deferred_checks);
    }

    for (const auto& entry : pending) {
        check_leaf_regions_depth2(*entry.tree, entry.dataview);
        int cpu_mis = cpu_mis_on_dataview(*entry.tree, entry.dataview);
        std::cout << "[CHECK] GPU_score=" << entry.tree->misclassification_score
                  << " CPU_eval=" << cpu_mis << "\n";
        RUNTIME_ASSERT(entry.tree->misclassification_score == cpu_mis,
                       "GPU misclassification_score != CPU evaluated mis");
    }
}

#endif
