// utils.cu
#include <cuda_runtime.h>
#include "utils.h"
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
    real *dst,
    const real *values,
    const int *col_idx)
{
    constexpr int blockSize = BLOCK_SIZE;
    int gridSize = (nnz + blockSize - 1) / blockSize;
    fill<real>(dst, 0.0, nnz);
    csr_column_sum_kernel<real><<<gridSize, blockSize>>>(nnz, dst, values, col_idx);
    cudaDeviceSynchronize();
}

template void csr_column_sum<double>(
    const int, double *, const double *, const int *);
// template void csr_column_sum<float>(
//     const int, float *, const float *, const int *);
template void csr_column_sum<int>(
    const int, int *, const int *, const int *);
