#include "fisher_func.h"
#include "utils.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
template <typename real>
__device__ real max(real a, real b)
{
    return (a > b) ? a : b;
}

template <typename real>
__device__ __forceinline__ real warpReduce(real val)
{
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ******************************************
// This Kernel is used to reduce the values in a block. Note that each block will have the duty of reducing 2 * blockSize values.
template <int blockSize, typename real>
__global__ void blockReduce(real *d_A, const int N)
{
    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    int i = 2 * blockSize * blockIdx.x + threadIdx.x;
    real sum = i < N ? d_A[i] : 0.0f;
    if (i + blockSize < N)
    {
        sum += d_A[i + blockSize];
    }
    shm[tid] = sum;
    __syncthreads();
    if (blockSize >= 1024 && tid < 512)
    {
        shm[tid] = sum = shm[tid] + shm[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512 && tid < 256)
    {
        shm[tid] = sum = shm[tid] + shm[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256 && tid < 128)
    {
        shm[tid] = sum = shm[tid] + shm[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128 && tid < 64)
    {
        shm[tid] = sum = shm[tid] + shm[tid + 64];
        __syncthreads();
    }
    if (blockSize >= 64 && tid < 32)
    {
        shm[tid] = sum = shm[tid] + shm[tid + 32];
        __syncthreads();
    }
    if (tid < 32)
    {
        sum = warpReduce<real>(sum);
    }
    if (tid == 0)
    {
        d_A[blockIdx.x] = sum; // 将结果存回全局内存
    }
}

template <int blockSize, typename real>
__global__ void gradient_csr_kernel(
    const int row,
    const real *__restrict__ x_val,
    const real *__restrict__ u_val,
    const real *__restrict__ w,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_indice,
    const real power,
    const real *__restrict__ p,
    const real tau,
    const real *__restrict__ x_old_val,
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
        real x_old = __ldg(&x_old_val[j]);
        real p_col = __ldg(&p[col_indice[j]]);

        real x_safe = fmaxf(x, 1e-12f);
        real x_pow = (power == 2) ? x_safe : powf(x_safe, power - 1);

        gradient[j] = -w_row * u * x_pow / utility_denom + p_col + tau * (x - x_old);
    }
}

template <typename real>
void launch_gradient_csr(
    const int row,
    const real *d_x_val,
    const real *d_u_val,
    const real *d_w,
    const int *d_row_ptr,
    const int *d_col_indice,
    const real power,
    const real *d_p,
    const real tau,
    const real *d_x_old_val,
    real *d_utility_no_power,
    real *d_gradient)
{
    const int blockSize = 256;
    const int gridSize = row; // 每行一个 block
    launch_utility_csr<real>(row, d_x_val, d_u_val, d_row_ptr, d_utility_no_power, power);
    cudaDeviceSynchronize();
    gradient_csr_kernel<blockSize, real><<<gridSize, blockSize, 0>>>(
        row, d_x_val, d_u_val, d_w, d_row_ptr, d_col_indice, power,
        d_p, tau, d_x_old_val, d_utility_no_power, d_gradient);
    cudaDeviceSynchronize();
}
template void launch_gradient_csr<double>(
    const int, const double *, const double *, const double *, const int *, const int *, const double, const double *, const double, const double *, double *, double *);
template void launch_gradient_csr<float>(
    const int, const float *, const float *, const float *, const int *, const int *, const float, const float *, const float, const float *, float *, float *);

template <int blockSize, typename real>
__global__ void objective_csr_kernel1(
    const int row,
    const real *__restrict__ x_val,
    const real *__restrict__ u_val,
    const real *__restrict__ w,
    const int *__restrict__ row_ptr,
    const real power,
    real *__restrict__ objective,
    const real tau,
    const real *__restrict__ x_old_val)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;

    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    int tid = threadIdx.x;

    __shared__ real shm_sum[blockSize];
    __shared__ real shm_ptot[blockSize];
    __shared__ real shm_delta[blockSize];

    real sum = 0.0;
    real x_delta = 0.0;

    for (int j = start + tid; j < end; j += blockDim.x)
    {
        real xi = __ldg(&x_val[j]);
        real x_old = __ldg(&x_old_val[j]);
        real ui = __ldg(&u_val[j]);

        real x_pow;
        if (power == 2)
            x_pow = xi * xi;
        else if (power == 3)
            x_pow = xi * xi * xi;
        else
            x_pow = powf(xi, power);

        sum += x_pow * ui;
        x_delta += (xi - x_old) * (xi - x_old);
    }

    shm_sum[tid] = sum;
    shm_delta[tid] = x_delta;
    __syncthreads();

    for (int offset = blockSize >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            shm_sum[tid] = sum = shm_sum[tid] + shm_sum[tid + offset];
            shm_delta[tid] = x_delta = shm_delta[tid] + shm_delta[tid + offset];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        sum = warpReduce<real>(sum);
        x_delta = warpReduce<real>(x_delta);
    }
    if (tid == 0)
    {
        real log_sum = logf(fmaxf(powf(sum, 1.0 / power), 1e-12f));
        objective[row_idx] = -w[row_idx] * log_sum + (0.5f * tau) * x_delta;
    }
}

template <int blockSize, typename real>
__global__ void objective_csr_kernel2(
    const int                col,        
    const real* __restrict__ x_sum,
    const real* __restrict__ p,
    const real* __restrict__ b,
    real*                    d_obj)        
{
    __shared__ real shm[blockSize];  
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    real partial = real(0);
    if (gid < col) {
        partial = p[gid] * (x_sum[gid] - b[gid]);
    }
    shm[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_obj, shm[0]);
    }
}
template <typename real>
void launch_objective_csr(
    const int row,
    const int col,
    const int nnz,
    const real *d_x_val,
    const real *d_u_val,
    const real *d_w,
    const int *d_row_ptr,
    const int *d_col_indice,
    const real power,
    real *d_objective,
    real &obj,
    real *x_sum,
    const real *d_p,
    const real tau,
    const real *d_x_old_val,
    const real *d_b)
{
    const int blockSize = 256;

    //=============Calculate Part. 1 \sum_{i=1}^{n} - w_i \log(\sum_{j=1}^{m} x_{ij}^{power} u_{ij}) + 1/2 \rho || x - x_{old} ||^2
    int gridSize = row;
    objective_csr_kernel1<256, real><<<gridSize, blockSize, 0>>>(
        row, d_x_val, d_u_val, d_w, d_row_ptr, power,
        d_objective, tau, d_x_old_val);
    cudaDeviceSynchronize();

    int numBlocks = row;
    for (numBlocks = row; numBlocks > 1; numBlocks = (numBlocks + blockSize - 1) / blockSize)
    {
        int gridSize = (numBlocks + 2*blockSize - 1) / (2*blockSize);
        blockReduce<blockSize, real><<<gridSize, blockSize, 0>>>(d_objective, numBlocks);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(&obj, d_objective, sizeof(real), cudaMemcpyDeviceToHost);
    //=============Calculate Part. 2 \sum_{j=1}^{m} p_j (\sum_{i=1}^{n} x_{ij}) - 1)
    csr_column_sum(nnz, col, x_sum, d_x_val, d_col_indice);
    int gridSize2 = (col + blockSize - 1) / blockSize;
    real* obj2 = nullptr;
    cudaMallocManaged(&obj2, sizeof(real));
    objective_csr_kernel2<blockSize, real>
        <<<gridSize2, blockSize>>>(col, x_sum, d_p, d_b, obj2);
    cudaDeviceSynchronize();
    real obj2_value;
    cudaMemcpy(&obj2_value, obj2, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(obj2);
    obj += obj2_value;
}
// 显式实例化常用类型
template void launch_objective_csr<double>(
    const int, const int, const int, const double *, const double *, const double *, const int *, const int *, const double, double *, double &, double *, const double *, const double, const double *, const double *);
// template void launch_objective_csr<float>(
//     int, int, int, const float *, const float *, const float *, const int *, const int *, const float, float *, float &, float *, const float *, const float, const float *);

template <int blockSize, typename real>
__global__ void utility_csr_kernel(
    const int row,
    const real *__restrict__ x_val,
    const real *__restrict__ u_val,
    const int *__restrict__ row_ptr,
    real *__restrict__ utility,
    const real power)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;
    // show blockIdx.x

    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];

    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    real sum = 0.0;

    for (int j = start + tid; j < end; j += blockDim.x)
    {
        real xi = x_val[j];
        real ui = u_val[j];

        sum += pow(xi, power) * ui; 
    }

    shm[tid] = sum;
    __syncthreads();

    for (int offset = blockSize >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            shm[tid] = sum = shm[tid] + shm[tid + offset];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        sum = warpReduce<real>(sum);
    }
    
    if (tid == 0)
    {
        utility[row_idx] = sum;
    }
}

template <typename real>
void launch_utility_csr(
    const int nnz,
    const real *d_x_val,
    const real *d_u_val,
    const int *d_row_ptr,
    real *d_utility,
    const real power)
{
    const int blockSize = 256; 
    int gridSize = row;     
    utility_csr_kernel<blockSize, real><<<gridSize, blockSize, 0>>>(
        row, d_x_val, d_u_val, d_row_ptr, d_utility, power);
    cudaDeviceSynchronize();
}

// 显式实例化常用类型
template void launch_utility_csr<double>(
    const int, const double *, const double *, const int *, double *, const double);
template void launch_utility_csr<float>(
    const int, const float *, const float *, const int *, float *, const float);

__global__ void calcualte_D_inv_d_hessian_CES_kernel(
    const int row,
    const double power,
    const double inv_tau,
    const double* __restrict__ d_w,
    const double* __restrict__ d_x_val,
    const int* __restrict__ d_row_ptr,
    const double* __restrict__ d_u_val,
    const double* __restrict__ d_utility_no_power,
    double* __restrict__ D_inv,
    double* __restrict__ d,
    double& D_inv_d)
{
    int row_idx = blockIdx.x;
    if (row_idx >= row)
        return;
    int start = d_row_ptr[row_idx];
    int end = d_row_ptr[row_idx + 1];
    double utility_denom = fmax(d_utility_no_power[row_idx], 1e-12);
    double w = d_w[row_idx];
    for (int j = start + threadIdx.x; j < end; j += blockDim.x)
    {
        double x = d_x_val[j];
        double u = d_u_val[j];
        
        D_inv_j = 1 / (inv_tau + (1.0 - power) * w * u * pow(x, power - 2) / utility_denom);
        d_j = sqrt(w * power) * u * pow(x, power - 1) / utility_denom;
        D_inv[j] = D_inv_j;
        d[j] = d_j;
        D_inv_d[j] = D_inv_j * d_j;
    }
    return;
}

    