// utils.cu
#include <cuda_runtime.h>
#include "utils.h"
#include <curand_kernel.h>
constexpr int BLOCK_SIZE = 256;

template <int blockSize, typename real>
__global__ void fill_kernel(real *dst, const real value, const int size)
{
    int idx = blockIdx.x * blockSize + threadIdx.x;
    if (idx < size)
        dst[idx] = value;
}

template <typename real>
void fill(real *dst, const real value, const int size)
{
    constexpr int blockSize = BLOCK_SIZE;
    int grid = (size + blockSize - 1) / blockSize;
    fill_kernel<blockSize><<<grid, blockSize>>>(dst, value, size);
    cudaDeviceSynchronize();
}
// 3. 最后显式实例化
template void fill<double>(double *, double, int);
template void fill<float>(float *, float, int);
template void fill<int>(int *, int, int);
template void fill<unsigned int>(unsigned int *, unsigned int, int);
template void fill<long long>(long long *, long long, int);
template void fill<unsigned long long>(unsigned long long *, unsigned long long, int);
template void fill<char>(char *, char, int);

// 全局状态数组
curandState *d_states = nullptr;
int d_states_size = 0;

__global__ void init_curand_states(curandState *states, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}


void init_global_curand(int size, unsigned long seed = 1234ULL)
{
    if (d_states && size == d_states_size) return;  // 已初始化
    cudaFree(d_states);
    d_states_size = size;
    cudaMalloc(&d_states, size * sizeof(curandState));

    int block = 256;
    int grid  = (size + block - 1) / block;
    init_curand_states<<<grid, block>>>(d_states, seed, size);
    cudaDeviceSynchronize();
}

template<typename real>
__global__ void uniform_rand_fill_core(curandState *states, real *dst,
                                       real low, real high, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    real val = curand_uniform(&states[idx]);
    dst[idx] = low + (high - low) * val;
}


void uniform_rand_fill(double *dst, double low, double high, int size)
{
    init_global_curand(size);                 // 只初始化一次
    int block = 256;
    int grid  = (size + block - 1) / block;
    uniform_rand_fill_core<<<grid, block>>>(d_states, dst, low, high, size);
    cudaDeviceSynchronize();
}


// __device__ inline void atomicAdd(float *address, float value)

// {
//     float old = value;
//     float new_old;
//     do
//     {
//         new_old = atomicExch(address, 0.0f);
//         new_old += old;
//     }
//     while ((old = atomicExch(address, new_old)) != 0.0f);
// };

template <typename real>
__global__ void csr_column_sum_kernel(
    const int nnz,
    real *dst,
    const real *values,
    const int *col_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz)
    {
        atomicAdd(&dst[col_idx[idx]], values[idx]);
    }
}

template <typename real>
void csr_column_sum(
    const int nnz,
    const int col_dim,
    real *dst,
    const real *values,
    const int *col_idx)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (nnz + blockSize - 1) / blockSize;
    fill<real>(dst, 0.0, col_dim);
    csr_column_sum_kernel<real><<<gridSize, blockSize>>>(nnz, dst, values, col_idx);
    cudaDeviceSynchronize();
}

template void csr_column_sum<double>(
    const int, const int, double *, const double *, const int *);
// template void csr_column_sum<float>(
//     const int, float *, const float *, const int *);
template void csr_column_sum<int>(
    const int, const int, int *, const int *, const int *);

template <typename real>
__global__ void weighted_self_add_kernel(
    const int N,
    real* dst,
    const real* src,
    const real weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] += src[idx] * weight;
    }
}

template <typename real>
void weighted_self_add(
    const int N,
    real* dst,
    const real* src,
    const real weight
){
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    weighted_self_add_kernel<real><<<gridSize, blockSize>>>(N, dst, src, weight);
    cudaDeviceSynchronize();
}

template void weighted_self_add<double>(const int, double*, const double*, const double);


template <typename real>
__global__ void element_a_minus_b_kernel(
    const int N,
    real* dst,
    const real* a,
    const real* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = a[idx] - b[idx];
    }
}
template <typename real>
void element_a_minus_b(
    const int N,
    real* dst,
    const real* a,
    const real* b)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    element_a_minus_b_kernel<real><<<gridSize, blockSize>>>(N, dst, a, b);
    cudaDeviceSynchronize();
}

template void element_a_minus_b<double>(
    const int N, double* dst, const double* a, const double* b);

template <typename real>
__global__ void weighted_self_add_diff_kernel(
    const int N,
    real* dst,
    const real* src,
    const real weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] += (src[idx] - dst[idx]) * weight;
    }
}
template <typename real>
void weighted_self_add_diff(
    const int N,
    real* dst,
    const real* src,
    const real weight
){
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    weighted_self_add_diff_kernel<real><<<gridSize, blockSize>>>(N, dst, src, weight);
    cudaDeviceSynchronize();
}

template void weighted_self_add_diff<double>(const int, double*, const double*, const double);
template void weighted_self_add_diff<float >(const int, float*,  const float*,  const float);

template <typename real>
__global__ void self_add_kernel(
    const int N,
    real* dst,
    const real* src
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] += src[idx];
    }
}

template <typename real>
// Adds src to dst element-wise
void self_add(
    const int N,
    real* dst,
    const real* src
)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    self_add_kernel<real><<<gridSize, blockSize>>>(N, dst, src);
    cudaDeviceSynchronize();
}

template <typename real>
__global__ void weighted_self_add_abs_kernel(
    const int N,
    real* dst,
    const real* src,
    const real weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = fabs(dst[idx] + src[idx] * weight);
    }
}

template <typename real>
void weighted_self_add_abs(
    const int N,
    real* dst,
    const real* src,
    const real weight
)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    weighted_self_add_abs_kernel<real><<<gridSize, blockSize>>>(N, dst, src, weight);
    cudaDeviceSynchronize();
}
template void weighted_self_add_abs<double>(int, double*, const double*, double);
template void weighted_self_add_abs<float >(int, float*,  const float*,  float);

template <typename real>
__global__ void abs_cuda_kernel(
    const int N,
    real* dst,
    const real* src
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = fabs(src[idx]);
    }
}

template <typename real>
void abs_cuda(
    const int N,
    real* dst,
    const real* src
)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    abs_cuda_kernel<real><<<gridSize, blockSize>>>(N, dst, src);
    cudaDeviceSynchronize();
}

template void abs_cuda<double>(int, double*, const double*);
template void abs_cuda<float >(int, float*,  const float*);

template <typename real>
__global__ void self_div_kernel(
    const int N,
    real* dst,
    const real* src
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (dst[idx] != 0) {
            dst[idx] = src[idx] / dst[idx]; // 避免除零
        } else { 
            dst[idx] = 0; // 避免除零
        }
    }
}

template <typename real>
void self_div(
    const int N,
    real* dst,
    const real* src
){
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    self_div_kernel<real><<<gridSize, blockSize>>>(N, dst, src);
    cudaDeviceSynchronize();
}

template void self_div<double>(const int, double*, const double*);
template void self_div<float >(const int, float*,  const float*);

template <typename real>
__global__ void detect_inf_nan_kernel(
    const real* x,
    int n,
    int* flag)  // 0 = ok, 1 = inf/nan found
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    real v = x[idx];
    if (isnan(v) || isinf(v)) {
        atomicOr(flag, 1);  // 原子设置标志位
    }
}
template <typename real>
bool detect_inf_nan(
    const real* x,
    int n)
{
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));  // 初始化标志位
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;
    detect_inf_nan_kernel<real><<<gridSize, blockSize>>>(x, n, d_flag);
    cudaDeviceSynchronize();
    int h_flag;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    return h_flag != 0;  // 如果标志位为1，表示有 inf 或 nan
}

template bool detect_inf_nan<double>(const double*, int);

