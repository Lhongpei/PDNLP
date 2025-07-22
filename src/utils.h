
#pragma once
#include <cuda_runtime.h>
template <typename real>
void fill(
    real* dst,
    const real value,
    const int size);

template <typename real>
void csr_column_sum(
    const int nnz,
    real* dst,
    const real *values,
    const int *col_idx
);

