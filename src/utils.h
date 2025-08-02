
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
template <typename real>
void fill(
    real* dst,
    const real value,
    const int size);
template <typename real>
void element_a_minus_b(
    const int N,
    real* dst,
    const real* a,
    const real* b);
template <typename real>
void csr_column_sum(
    const int nnz,
    const int col_dim,
    real* dst,
    const real *values,
    const int *col_idx
);
template <typename real>
void weighted_self_add(
    const int N,
    real* dst,
    const real* src,
    const real weight
);

template <typename real>
void weighted_self_add_diff(
    const int N,
    real* dst,
    const real* src,
    const real weight
);

template <typename real>
void self_add(
    const int N,
    real* dst,
    const real* src
);

template <typename real>
void self_div(
    const int N,
    real* dst,
    const real* src
);

template <typename real>
void weighted_self_add_abs(
    const int N,
    real* dst,
    const real* src,
    const real weight
);

template <typename real>
void abs_cuda(
    const int N,
    real* dst,
    const real* src
);

void uniform_rand_fill(double *dst, double low, double high, int size);

//Designed for debugging purpose, it will copy the device array to host and print it.
template <typename real>
void print_cuarray(
    const char* name,
    const real* array,
    const int size
){
    real* h_array = new real[size];
    cudaMemcpy(h_array, array, size * sizeof(real), cudaMemcpyDeviceToHost);
    printf("%s: ", name);
    for (int i = 0; i < size; ++i) {
        //check if the type is integer
        if constexpr (std::is_integral<real>::value) {
            printf("%d ", static_cast<int>(h_array[i]));
        } else {
            printf("%f ", h_array[i]);
        }
    }
    printf("\n");
    delete[] h_array;
}

#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

template <typename real>
bool detect_inf_nan(
    const real* x,
    int n);