#include <cuda_runtime.h>
#include "fisher_dual.h"

template <typename real>
__global__ void fisher_dual_kernel(
    const int n,
    real* __restrict__ dual_solution,
    const real* __restrict__ last_dual_solution,
    const real* __restrict__ primal_sum,
    const real* __restrict__ last_primal_sum,
    const real* __restrict__ right_hand_side,
    const real sigma
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dual_solution[idx] = last_dual_solution[idx] + sigma * (2 * primal_sum[idx] - last_primal_sum[idx] - right_hand_side[idx]);
    }
}

// struct alignas(16) Pack {
//     double last_dual, primal, last_primal, rhs;
// };

// __global__ void update_dual(Pack *pack,
//                             double *dual,
//                             double sigma,
//                             int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         auto p = pack[idx];
//         dual[idx] = p.last_dual +
//                     sigma * (2.0 * p.primal - p.last_primal - p.rhs);
//     }
// }