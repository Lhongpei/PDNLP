#pragma once
#include <cuda_runtime.h>

struct FisherProblem {
    double *x0, *w, *u_val, *b;
    int    *col_ind, *row_ptr;
    double *bounds;   // 3 x nnz
    double power;  // 用于计算效用和目标函数的幂
};

void generate_problem_gpu(int m, int n, int nnz, FisherProblem &csr, double power = 1.0);