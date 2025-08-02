#include <cuda_runtime.h>
#include "fisher_dual.h"

template <typename real>
__global__ void fisher_dual_kernel(
    const int col,
    real* __restrict__ dual_solution,
    const real* __restrict__ last_dual_solution,
    const real* __restrict__ primal_sum,
    const real* __restrict__ last_primal_sum,
    const real* __restrict__ right_hand_side,
    const real sigma
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < col) {
        dual_solution[idx] = last_dual_solution[idx] + sigma * (2 * primal_sum[idx] - last_primal_sum[idx] - right_hand_side[idx]);
        // printf("Check all components: dual_solution[%d] = %f, last_dual_solution[%d] = %f, primal_sum[%d] = %f, last_primal_sum[%d] = %f, right_hand_side[%d] = %f\n",
        //        idx, dual_solution[idx], idx, last_dual_solution[idx], idx, primal_sum[idx], idx, last_primal_sum[idx], idx, right_hand_side[idx]);
    }
}

template <typename real>
void dual_update(
    const int col,
    real* dual_solution,
    const real* last_dual_solution,
    const real* primal_sum,
    const real* last_primal_sum,
    const real* right_hand_side,
    const real sigma
){
    constexpr int blockSize = 256;
    int gridSize = (col + blockSize - 1) / blockSize;
    fisher_dual_kernel<real><<<gridSize, blockSize>>>(
        col, dual_solution, last_dual_solution, primal_sum, last_primal_sum, right_hand_side, sigma);
    cudaDeviceSynchronize();
}
// 显式实例化常用类型
template void dual_update<double>(
    const int, double *, const double *, const double *, const double *, const double *, const double);
template void dual_update<float>(
    const int, float *, const float *, const float *, const float *, const float *, const float);

// struct alignas(16) Pack {
//     double last_dual, primal, last_primal, rhs;
// };

// __global__ void update_dual(Pack *pack,
//                             double *dual,
//                             double sigma,
//                             int col)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < col) {
//         auto p = pack[idx];
//         dual[idx] = p.last_dual +
//                     sigma * (2.0 * p.primal - p.last_primal - p.rhs);
//     }
// }