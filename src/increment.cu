#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <tuple>
#include <cmath>
#include "increment.h"
#include "reduction.h"
#include "utils.h"
#include "const.h"
#include "fisher_func.h"
#include <printf.h>
#include <iostream>
#include "const.h"
#include <assert.h>
template <typename real>
std::tuple<real, real> compute_interaction_and_movement(
    int        col,
    int        nnz,
    real       primal_weight,
    real*      d_current_primal_solution,
    real*      d_last_primal_solution,
    real*      d_current_dual_solution,
    real*      d_last_dual_solution,
    real*      d_current_primal_sum,
    real*      d_last_primal_sum,
    real*      cache_delta_primal,
    real*      cache_delta_dual,
    real*      cache_delta_primal_sum,
    cublasHandle_t handle)
{
    element_a_minus_b(nnz, cache_delta_primal, d_current_primal_solution, d_last_primal_solution);
    element_a_minus_b(col, cache_delta_dual, d_current_dual_solution, d_last_dual_solution);
    element_a_minus_b(col, cache_delta_primal_sum, d_current_primal_sum, d_last_primal_sum);

    real interaction = 0.0;
    real *d_interaction = nullptr;
    cudaMalloc(&d_interaction, sizeof(real));

    cublasStatus_t status = cublasDdot(handle, col,
            cache_delta_primal_sum, 1,
            cache_delta_dual,       1,
            d_interaction); // Now cuBLAS correctly writes to this host variable
    //Check for errors
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS error: " << status << std::endl;
        //Stop the program immediately
        cudaFree(d_interaction);
        throw std::runtime_error("CUBLAS error in cublasDdot");
    }
    
    real h_interaction;
    cudaMemcpy(&h_interaction, d_interaction, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_interaction);
    interaction = fabs(h_interaction);

    // 4. norms
    real norm_dp = 0.0, norm_dd = 0.0;
    real *d_norm_dp = nullptr;
    real *d_norm_dd = nullptr;
    cudaMalloc(&d_norm_dp, sizeof(real));
    cudaMalloc(&d_norm_dd, sizeof(real));
    cublasDdot(handle, nnz, cache_delta_primal, 1, cache_delta_primal, 1, d_norm_dp);
    cublasDdot(handle, col, cache_delta_dual,   1, cache_delta_dual,   1, d_norm_dd);
    real h_norm_dp, h_norm_dd;
    cudaMemcpy(&h_norm_dp, d_norm_dp, sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_norm_dd, d_norm_dd, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_norm_dp);
    cudaFree(d_norm_dd);
    cudaDeviceSynchronize();
    // norm_dp = sqrt(norm_dp);
    // norm_dd = sqrt(norm_dd);

    // 5. movement = 0.5*(‖Δp‖² + ‖Δd‖²)
    real movement = 0.5 * (primal_weight *  h_norm_dp + 1.0/ primal_weight * h_norm_dd);


    return std::make_tuple(interaction, movement);
}
template std::tuple<double, double> compute_interaction_and_movement(int, int, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, cublasHandle_t);

template<typename real>
real cal_feasibility(
    int N,
    const real* x_sum,
    const real* b,
    bool relative
){
    real norm = 0.0;
    real *tmp;
    cudaMalloc(&tmp, N * sizeof(real));
    cudaMemcpy(tmp, x_sum, N * sizeof(real), cudaMemcpyDeviceToDevice);
    weighted_self_add(N, tmp, b, -1.0);
    norm = cuNorm(tmp, N, NORM_INF);
    if (!relative) return norm;
    real norm_x = cuNorm(x_sum, N, NORM_INF);
    abs_cuda(N, tmp, b);
    real norm_b = cuNorm(b, N, NORM_INF);
    cudaFree(tmp);
    // printf("norm_x = %f, norm_b = %f, norm = %f\n", norm_x, norm_b, norm);
    return norm / (1 + std::max(norm_x, norm_b));
}

template double cal_feasibility(int, const double*, const double*, bool);


template <typename real>
__device__ __forceinline__ real max(real a, real b)
{
    return (a > b) ? a : b;
}


template<typename real>
__global__ void kernel1(
    int N,
    const real* d_in1,
    const real* d_in2,
    real* d_out
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = max(0.0, d_in1[idx] ) * d_in2[idx];
    }
}

template <typename real>
real cal_gap(
    int nnz,
    const real* primal_solution,
    const real* primal_gradient,
    bool relative
)
{
    real *tmp;
    cudaMalloc(&tmp, nnz * sizeof(real));
    kernel1<<<(nnz + 255) / 256, 256>>>(
        nnz,
        primal_solution,
        primal_gradient,
        tmp
    );
    real abs_gap = cuNorm(tmp, nnz, NORM_INF);
    if (!relative) {
        cudaFree(tmp);
        return abs_gap;
    }
    real norm_x = cuNorm(primal_solution, nnz, NORM_INF);
    real norm_g = cuNorm(primal_gradient, nnz, NORM_INF);
    cudaFree(tmp);
    return abs_gap / (1 + max(norm_x, norm_g));
}
template double cal_gap(int, const double*, const double*, bool);

template <int blockSize, typename real>
__global__ void gradient_original(
    const int row,
    const real *__restrict__ x_val,
    const real *__restrict__ u_val,
    const real *__restrict__ w,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_indice,
    const real power,
    const real *__restrict__ p,
    const real *__restrict__ utility_no_power,
    real *__restrict__ gradient)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;
    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    int tid = threadIdx.x;

    real utility_denom = fmaxf(utility_no_power[row_idx], 1e-12f);
    real w_row = w[row_idx];

#pragma unroll 4
    for (int j = start + tid; j < end; j += blockDim.x)
    {
        real x = __ldg(&x_val[j]);
        real u = __ldg(&u_val[j]);
        real p_col = __ldg(&p[col_indice[j]]);

        real x_safe = fmaxf(x, 1e-12f);
        real x_pow = (power == 2) ? x_safe : powf(x_safe, power - 1); // 特化 power=2 的情况

        gradient[j] = -w_row * u * x_pow / utility_denom + p_col;
    }
}

template <int blockSize, typename real>
__global__ void gradient_original_part1(
    const int row,
    const real *__restrict__ x_val,
    const real *__restrict__ u_val,
    const real *__restrict__ w,
    const int *__restrict__ row_ptr,
    const real power,
    const real *__restrict__ utility_no_power,
    real *__restrict__ gradient)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;
    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    int tid = threadIdx.x;

    real utility_denom = fmaxf(utility_no_power[row_idx], static_cast<real>(EPS));
    real w_row = w[row_idx];

#pragma unroll 4
    for (int j = start + tid; j < end; j += blockDim.x)
    {
        real x = __ldg(&x_val[j]);
        real u = __ldg(&u_val[j]);

        real x_safe = fmaxf(x, 1e-12f);
        real x_pow = (power == 2) ? x_safe : powf(x_safe, power - 1); // 特化 power=2 的情况

        gradient[j] = -w_row * u * x_pow / utility_denom ;
    }
}
template <int blockSize, typename real>
__global__ void gradient_original_part2(
    const int row,
    real *__restrict__ gradient_cache,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_indice,
    const real *__restrict__ p
)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;
    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    int tid = threadIdx.x;
    for (int j = start + tid; j < end; j += blockDim.x)
    {
        real p_col = __ldg(&p[col_indice[j]]);
        gradient_cache[j] += p_col;
    }
}
template <typename real>
real cal_dual_res(
    const int row_dim,
    const int col_dim,
    const int nnz,
    const real* w,
    const real* x_val,
    const real* u_val,
    const int* row_ptr,
    const int* col_ind,
    const real* p,
    const real power,
    bool relative
){
    real *utility;
    cudaMalloc(&utility, row_dim * sizeof(real));
    cudaMemset(utility, 0, row_dim * sizeof(real));
    launch_utility_csr(
        row_dim, x_val, u_val, row_ptr, utility, power);
    if (!relative) {
        real *tmp;
        cudaMalloc(&tmp, nnz * sizeof(real));
        gradient_original<256, real><<<row_dim, 256>>>(
            row_dim, x_val, u_val, w, row_ptr, col_ind, power, p, utility, tmp);
        real abs_res = cuNorm(tmp, nnz, NORM_INF);
        cudaFree(tmp);
        cudaFree(utility);
        return abs_res;
    } else {
        real *tmp;
        cudaMalloc(&tmp, nnz * sizeof(real));
        gradient_original_part1<256, real><<<row_dim, 256>>>(
            row_dim, x_val, u_val, w, row_ptr, power, utility, tmp);
        real norm_part1 = cuNorm(tmp, nnz, NORM_INF);
        gradient_original_part2<256, real><<<row_dim, 256>>>(
            row_dim, tmp, row_ptr, col_ind, p);
        real norm_part2 = cuNorm(tmp, nnz, NORM_INF);
        cudaFree(tmp);
        real norm_part3 = cuNorm(p, col_dim, NORM_INF);
        // std::cout << "norm_part1: " << norm_part1
        //           << ", norm_part2: " << norm_part2
        //           << ", norm_part3: " << norm_part3 << std::endl;
        cudaFree(utility);
        return norm_part2 / (1 + max(norm_part1, norm_part3));
        }
    
}

template double cal_dual_res(
    const int, const int, const int,
    const double*, const double*,
    const double*, const int*, const int*, const double*, const double, bool);

template <typename real>
real compute_new_primal_weight(
    const real primal_distance,
    const real dual_distance,
    const real primal_weight,
    const real primal_weight_update_smoothing
){
    if (primal_distance > static_cast<real>(EPS) && dual_distance > static_cast<real>(EPS)) {
        real new_estimate = dual_distance / primal_distance;
        real log_estimate = primal_weight_update_smoothing * log(new_estimate) 
                            + (1 - primal_weight_update_smoothing) * log(primal_weight);
        real new_weight = exp(log_estimate);
        return new_weight;
    }
    return primal_weight;
}
template double compute_new_primal_weight(
    const double, const double, const double, const double);
    