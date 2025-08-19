#include <cooperative_groups.h>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm> 
#include <cmath>
#include "utils.h"
// Enum to specify the reduction operation
#include "reduction.h"

/**
 * @brief Performs a reduction within a single warp.
 * This version is templated on the ReductionOp to generate optimal code.
 */
template <typename real>
__device__ __forceinline__ real warpReduceSum(real val)
{
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

template <typename real> 
__device__ __forceinline__ real warpReduceMin(real val)
{
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

template <typename real>
__device__ __forceinline__ real warpReduceMax(real val)
{
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

template <ReductionOp op,typename real>
__device__ __forceinline__ real OPT(real val1, real val2)
{
    if constexpr (op == OP_SUM) {
        return val1 + val2;
    } else if constexpr (op == OP_MIN) {
        return fminf(val1, val2);
    } else if constexpr (op == OP_MAX) {
        return fmaxf(val1, val2);
    }
    return 0; // Should never reach here
}
/**
 * @brief The main reduction kernel. Reduces d_in of size N into d_out of size gridDim.x.
 * This kernel can be called recursively.
 */

template <int blockSize, ReductionOp op, typename real>
__global__ void reductionKernel(const real* __restrict__ d_in, real* __restrict__ d_out, const int N)
{
    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    int i = 2 * blockSize * blockIdx.x + threadIdx.x;
    real sum;
    if constexpr (op == OP_SUM) {
        sum = (i < N) ? d_in[i] : 0.0f;
    } else if constexpr (op == OP_MIN) {
        sum = (i < N) ? d_in[i]: INFINITY;
    } else if constexpr (op == OP_MAX) {
        sum = (i < N) ? d_in[i] : -INFINITY;
    }
    if (i + blockSize < N)
    {
        sum = OPT<op>(sum, d_in[i + blockSize]);
    }
    shm[tid] = sum;
    __syncthreads();
    if (blockSize >= 1024 && tid < 512)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 512]);
        __syncthreads();
    }
    if (blockSize >= 512 && tid < 256)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256 && tid < 128)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128 && tid < 64)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 64]);
        __syncthreads();
    }
    if (blockSize >= 64 && tid < 32)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 32]);
        __syncthreads();
    }
    if (tid < 32)
    {
        if constexpr (op == OP_SUM) {
            sum = warpReduceSum(sum);
        } else if constexpr (op == OP_MIN) {
            sum = warpReduceMin(sum);
        } else if constexpr (op == OP_MAX) {
            sum = warpReduceMax(sum);
        }
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = sum; 
    }
}
template <typename real>
struct Abs  { __device__ __host__ real operator()(real x) const { return fabsf(x); } };

template <typename real>
struct Sqr  { __device__ __host__ real operator()(real x) const { return x * x; } };

template <int blockSize, ReductionOp op, typename real, typename Func>
__global__ void reductionKernelwithFunc(const real* __restrict__ d_in, real* __restrict__ d_out, const int N)
{
    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    int i = 2 * blockSize * blockIdx.x + threadIdx.x;
    real sum;
    if constexpr (op == OP_SUM) {
        sum = (i < N) ? Func{}(d_in[i]) : 0.0f;
    } else if constexpr (op == OP_MIN) {
        sum = (i < N) ? Func{}(d_in[i]): INFINITY;
    } else if constexpr (op == OP_MAX) {
        sum = (i < N) ? Func{}(d_in[i]) : -INFINITY;
    }
    if (i + blockSize < N)
    {
        sum = OPT<op>(sum, Func{}(d_in[i + blockSize]));
    }
    shm[tid] = sum;
    __syncthreads();
    if (blockSize >= 1024 && tid < 512)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 512]);
        __syncthreads();
    }
    if (blockSize >= 512 && tid < 256)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256 && tid < 128)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128 && tid < 64)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 64]);
        __syncthreads();
    }
    if (blockSize >= 64 && tid < 32)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 32]);
        __syncthreads();
    }
    if (tid < 32)
    {
        if constexpr (op == OP_SUM) {
            sum = warpReduceSum(sum);
        } else if constexpr (op == OP_MIN) {
            sum = warpReduceMin(sum);
        } else if constexpr (op == OP_MAX) {
            sum = warpReduceMax(sum);
        }
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = sum;
    }
}
template <int blockSize, ReductionOp op, typename real>
__global__ void reductionSelfKernel(real* __restrict__ data, const int N)
{
    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    int i = 2 * blockSize * blockIdx.x + threadIdx.x;
    real sum;
    if constexpr (op == OP_SUM) {
        sum = (i < N) ? data[i] : 0.0f;
    } else if constexpr (op == OP_MIN) {
        sum = (i < N) ? data[i]: INFINITY;
    } else if constexpr (op == OP_MAX) {
        sum = (i < N) ? data[i] : -INFINITY;
    }
    if (i + blockSize < N)
    {
        sum = OPT<op>(sum, data[i + blockSize]);
    }
    shm[tid] = sum;
    __syncthreads();
    if (blockSize >= 1024 && tid < 512)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 512]);
        __syncthreads();
    }
    if (blockSize >= 512 && tid < 256)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256 && tid < 128)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128 && tid < 64)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 64]);
        __syncthreads();
    }
    if (blockSize >= 64 && tid < 32)
    {
        shm[tid] = sum = OPT<op>(shm[tid], shm[tid + 32]);
        __syncthreads();
    }
    if (tid < 32)
    {
        if constexpr (op == OP_SUM) {
            sum = warpReduceSum(sum);
        } else if constexpr (op == OP_MIN) {
            sum = warpReduceMin(sum);
        } else if constexpr (op == OP_MAX) {
            sum = warpReduceMax(sum);
        }
    }
    if (tid == 0)
    {
        data[blockIdx.x] = sum;
    }
}


// Error checking macro

/**
 * @brief Host-side wrapper to perform a global reduction.
 * * @param d_input Pointer to the input data on the device.
 * @param N Number of elements in the input array.
 * @param op The reduction operation to perform (OP_SUM, OP_MIN, OP_MAX).
 * @return The final reduced value (sum, min, or max).
 */

template <typename real>
real globalDiffReduce(const real* d_1, const real* d_2, const int N, ReductionOp op, bool use_abs)
{
    if (N == 0) return 0; // Or appropriate identity for min/max
    if (N == 1) {
        real result;
        CUDA_CHECK(cudaMemcpy(&result, d_1, sizeof(real), cudaMemcpyDeviceToHost));
        weighted_self_add_abs(N, &result, d_2, -1.0); // result = d_1 - d_2
        return result;
    }

    constexpr int blockSize = 256; // A good default block size
    int gridSize = (N + 2 * blockSize - 1) / (2 * blockSize);

    // Allocate intermediate buffer
    real* d_inter;
    CUDA_CHECK(cudaMalloc(&d_inter, N * sizeof(real)));
    cudaMemcpy(d_inter, d_1, N * sizeof(real), cudaMemcpyDeviceToDevice);
    if (use_abs) {
        weighted_self_add_abs(N, d_inter, d_2, -1.0); // d_inter = d_1 - d_2
    } else {
        weighted_self_add(N, d_inter, d_2, -1.0); // d_inter = d_1 - d_2
    }
    int current_N = N;
    // Launch the first reduction pass
    switch(op) {
        case OP_SUM: 
            reductionSelfKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_inter, N);
            break;
        case OP_MIN:
            reductionSelfKernel<blockSize, OP_MIN, real><<<gridSize, blockSize>>>(d_inter, N);
            break;
        case OP_MAX:
            reductionSelfKernel<blockSize, OP_MAX, real><<<gridSize, blockSize>>>(d_inter, N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    current_N = gridSize;

    // Recursively call the kernel until only one element is left
    while (current_N > 1) {
        gridSize = (current_N + 2 * blockSize - 1) / (2 * blockSize);
        
        // If the next output would be larger than our intermediate buffer, we need another buffer.
        // For simplicity, we can just use the original input buffer if it's not the final step.
        // A more robust solution might use two intermediate buffers and swap between them.
        // Here, we just point output to a new space or reuse `d_inter`.

        switch(op) {
            case OP_SUM: 
                reductionSelfKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_inter, current_N);
                break;
            case OP_MIN:
                reductionSelfKernel<blockSize, OP_MIN, real><<<gridSize, blockSize>>>(d_inter, current_N);
                break;
            case OP_MAX:
                reductionSelfKernel<blockSize, OP_MAX, real><<<gridSize, blockSize>>>(d_inter, current_N);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        current_N = gridSize;
    }

    // Copy the final result back to the host
    real result;
    CUDA_CHECK(cudaMemcpy(&result, d_inter, sizeof(real), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_inter));
    
    return result;
}

template double globalDiffReduce(const double* d_1, const double* d_2, const int N, ReductionOp op, bool use_abs);

template<typename real>
real globalReduce(const real* d_input, const int N, ReductionOp op)
{
    if (N == 0) return 0; // Or appropriate identity for min/max
    if (N == 1) {
        real result;
        CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(real), cudaMemcpyDeviceToHost));
        return result;
    }

    constexpr int blockSize = 256; // A good default block size
    int gridSize = (N + 2 * blockSize - 1) / (2 * blockSize);

    // Allocate intermediate buffer
    real* d_inter;
    CUDA_CHECK(cudaMalloc(&d_inter, gridSize * sizeof(real)));

    // Pointers for recursive calls
    const real* d_in_ptr = d_input;
    real* d_out_ptr = d_inter;
    int current_N = N;

    // Launch the first reduction pass
    switch(op) {
        case OP_SUM: 
            reductionKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
        case OP_MIN:
            reductionKernel<blockSize, OP_MIN, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
        case OP_MAX:
            reductionKernel<blockSize, OP_MAX, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    current_N = gridSize;

    // Recursively call the kernel until only one element is left
    while (current_N > 1) {
        d_in_ptr = d_out_ptr; // The output of the last pass is the input for the next
        gridSize = (current_N + 2 * blockSize - 1) / (2 * blockSize);
        
        // If the next output would be larger than our intermediate buffer, we need another buffer.
        // For simplicity, we can just use the original input buffer if it's not the final step.
        // A more robust solution might use two intermediate buffers and swap between them.
        // Here, we just point output to a new space or reuse `d_inter`.
        if (gridSize > 1) {
            d_out_ptr = d_inter; // Can safely reuse the intermediate buffer
        } else {
            // This is the last pass, we can write the final result to the start of the buffer.
            d_out_ptr = d_inter; 
        }

        switch(op) {
            case OP_SUM: 
                reductionKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
            case OP_MIN:
                reductionKernel<blockSize, OP_MIN, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
            case OP_MAX:
                reductionKernel<blockSize, OP_MAX, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        current_N = gridSize;
    }

    // Copy the final result back to the host
    real result;
    CUDA_CHECK(cudaMemcpy(&result, d_out_ptr, sizeof(real), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_inter));
    
    return result;
}

template double globalReduce(const double* d_input, const int N, ReductionOp op);



template<typename real>
real cuNorm(const real* d_input, const int N, NormType norm_type)
{
    if (N == 0) return 0; // Or appropriate identity for min/max
    if (N == 1) {
        real result;
        CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(real), cudaMemcpyDeviceToHost));
        return result;
    }
    constexpr int blockSize = 256;
    int gridSize = (N + 2 * blockSize - 1) / (2 * blockSize);

    // Allocate intermediate buffer
    real* d_inter;
    CUDA_CHECK(cudaMalloc(&d_inter, gridSize * sizeof(real)));
    // Pointers for recursive calls
    const real* d_in_ptr = d_input;
    real* d_out_ptr = d_inter;
    int current_N = N;

    // Launch the first reduction pass

    switch (norm_type) {
        case NORM_L1: 
            reductionKernelwithFunc<blockSize, OP_SUM, real, Abs<real>><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
        case NORM_L2:
            reductionKernelwithFunc<blockSize, OP_SUM, real, Sqr<real>><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
        case NORM_INF:
            reductionKernelwithFunc<blockSize, OP_MAX, real, Abs<real>><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
            break;
        default:
            std::cerr << "Unsupported norm type!" << std::endl;
            CUDA_CHECK(cudaFree(d_inter));
            return 0; // Or appropriate identity for unsupported norm
    }

    CUDA_CHECK(cudaGetLastError());
    current_N = gridSize;

    // Recursively call the kernel until only one element is left
    while (current_N > 1) {
        d_in_ptr = d_out_ptr; // The output of the last pass is the input for the next
        gridSize = (current_N + 2 * blockSize - 1) / (2 * blockSize);
        
        // If the next output would be larger than our intermediate buffer, we need another buffer.
        // For simplicity, we can just use the original input buffer if it's not the final step.
        // A more robust solution might use two intermediate buffers and swap between them.
        // Here, we just point output to a new space or reuse `d_inter`.
        if (gridSize > 1) {
            d_out_ptr = d_inter; // Can safely reuse the intermediate buffer
        } else {
            // This is the last pass, we can write the final result to the start of the buffer.
            d_out_ptr = d_inter; 
        }

        switch (norm_type) {
            case NORM_L1: 
                reductionKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
            case NORM_L2:
                reductionKernel<blockSize, OP_SUM, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
            case NORM_INF:
                reductionKernel<blockSize, OP_MAX, real><<<gridSize, blockSize>>>(d_in_ptr, d_out_ptr, current_N);
                break;
            default:
                std::cerr << "Unsupported norm type!" << std::endl;
                CUDA_CHECK(cudaFree(d_inter));
                return 0; // Or appropriate identity for unsupported norm
        }
        CUDA_CHECK(cudaGetLastError());
        current_N = gridSize;
    }
    CUDA_CHECK(cudaGetLastError());
    // Copy the final result back to the host
    real result;
    CUDA_CHECK(cudaMemcpy(&result, d_out_ptr, sizeof(real), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_inter));
    if (norm_type == NORM_L2) {
        result = std::sqrt(result); // Take square root for L2 norm
    }
    
    return result;
}
template double cuNorm(const double* d_input, const int N, NormType norm_type);

// // Benchmarking and testing the reduction functions
// int main()
// {
//     const int N = 1 << 22; // ~16 million elements
//     std::vector<double> h_data(N);
//     double init_value = 1.0; // Initial value for the data
//     for (int i = 0; i < N; ++i) {
//         init_value *= -1.0; // Alternate sign for demonstration
//         h_data[i] =  init_value * static_cast<double>(i % 79)/10; // Some sample data
//     }

//     double* d_data;
//     CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(double)));
//     CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice));

//     // --- Perform Reductions ---
//     double sum_val = globalReduce(d_data, N, OP_SUM);
//     double min_val = globalReduce(d_data, N, OP_MIN);
//     double max_val = globalReduce(d_data, N, OP_MAX);
    
//     std::cout.precision(10);
//     std::cout << "GPU Sum: " << sum_val << std::endl;
//     std::cout << "GPU Min: " << min_val << std::endl;
//     std::cout << "GPU Max: " << max_val << std::endl;

//     // --- Verification on CPU ---
//     double cpu_sum = 0.0;
//     double cpu_min = std::numeric_limits<double>::infinity();
//     double cpu_max = -std::numeric_limits<double>::infinity();
//     for(int i=0; i<N; ++i) cpu_sum += h_data[i];
//     std::cout << "CPU Sum: " << cpu_sum << std::endl;
//     for(int i=0; i<N; ++i) {
//         if (h_data[i] < cpu_min) cpu_min = h_data[i];
//         if (h_data[i] > cpu_max) cpu_max = h_data[i];
//     }
//     std::cout << "CPU Min: " << cpu_min << std::endl;
//     std::cout << "CPU Max: " << cpu_max << std::endl;
    
//     double* d_data_2;
//     CUDA_CHECK(cudaMalloc(&d_data_2, N * sizeof(double)));
//     fill(d_data_2, 1.0, N); // Fill with 1.0 for testing globalDiffReduce
//     double diff_sum = globalDiffReduce(d_data, d_data_2, N, OP_SUM, true);
//     double diff_min = globalDiffReduce(d_data, d_data_2, N, OP_MIN, true);
//     double diff_max = globalDiffReduce(d_data, d_data_2, N, OP_MAX, true);
//     std::cout << "GPU Diff Sum: " << diff_sum << std::endl;
//     std::cout << "GPU Diff Min: " << diff_min << std::endl;
//     std::cout << "GPU Diff Max: " << diff_max << std::endl;

//     // Verification on CPU for globalDiffReduce
//     double cpu_diff_sum = 0.0;
//     double cpu_diff_min = std::numeric_limits<double>::infinity();
//     double cpu_diff_max = -std::numeric_limits<double>::infinity();
//     for(int i=0; i<N; ++i) {
//         double diff = abs(h_data[i] - 1.0); // Since d_data_2 is filled with 1.0
//         cpu_diff_sum += diff;
//         if (diff < cpu_diff_min) cpu_diff_min = diff;
//         if (diff > cpu_diff_max) cpu_diff_max = diff;
//     }
//     std::cout << "CPU Diff Sum: " << cpu_diff_sum << std::endl;
//     std::cout << "CPU Diff Min: " << cpu_diff_min << std::endl;
//     std::cout << "CPU Diff Max: " << cpu_diff_max << std::endl;
//     CUDA_CHECK(cudaFree(d_data));
//     return 0;
// }

// // template<typename real>
// // real cpuNorm(const std::vector<real>& v, NormType t)
// // {
// //     switch (t) {
// //         case NORM_L1:   { real s = 0; for (auto x : v) s += std::fabs(x); return s; }
// //         case NORM_L2:   { real s = 0; for (auto x : v) s += x * x;      return std::sqrt(s); }
// //         case NORM_INF:  { real m = 0; for (auto x : v) m = std::max(m, std::fabs(x)); return m; }
// //         default: return 0;
// //     }
// // }

// // // int main()
// // // {
// // //     using real = double; // Change to double if needed
// // //     const int N = 10007;                 
// // //     std::vector<real> h_in(N);
// // //     for (int i = 0; i < N; ++i) h_in[i] = real(i - 5000) * 0.123f; 

// // //     real *d_in;
// // //     cudaMalloc(&d_in, N * sizeof(real));
// // //     cudaMemcpy(d_in, h_in.data(), N * sizeof(real), cudaMemcpyHostToDevice);

// // //     const NormType types[] = { NORM_L1, NORM_L2, NORM_INF };
// // //     const char* names[]    = { "L1", "L2", "INF" };

// // //     for (int k = 0; k < 3; ++k)
// // //     {
// // //         real gpu = cuNorm<real>(d_in, N, types[k]); 

// // //         real cpu = cpuNorm(h_in, types[k]);

// // //         std::cout << names[k] << " norm: GPU = " << gpu
// // //                   << ", CPU = " << cpu
// // //                   << ", diff = " << std::fabs(gpu - cpu) << '\n';
// // //     }

// // //     cudaFree(d_in);
// // //     return 0;
// // // }
