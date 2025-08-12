#pragma once
#include <cuda_runtime.h>
#include "utils.h"
struct FisherProblem {
    double *x0, *w, *u_val, *b;
    int    *col_ind, *row_ptr;
    double *bounds;   // 3 x nnz
    double power;  // 用于计算效用和目标函数的幂
    int row_dim;  // 行数
    int col_dim;  // 列数
    int nnz;    // 非零元素个数
};
struct FisherProblemHost {
    int row_dim, col_dim, nnz;
    double power;
    std::vector<double> x0, w, u_val, b;
    std::vector<int> col_ind, row_ptr;
};
void print_fisher_problem(const FisherProblem &problem);

void generate_problem_gpu(int row, int col, int nnz, FisherProblem &csr, double power = 1.0, double b_value = 1.0);

FisherProblemHost to_host(const FisherProblem& gpu_problem);
void save_problem_to_files(const FisherProblemHost& prob,
                           const std::string& base_dir,
                           const std::string& stem);