#pragma once
#include <cuda_runtime.h>
#include "utils.h"
struct FisherProblem{
    double *x0, *w, *u_val, *b;
    int    *col_ind, *row_ptr;
    double *bounds;   
    double power;  
    int row_dim; 
    int col_dim;  
    int nnz; 
    FisherProblem()
        : x0(nullptr), w(nullptr), u_val(nullptr), b(nullptr),
          col_ind(nullptr), row_ptr(nullptr), bounds(nullptr),
          power(1.0), row_dim(0), col_dim(0), nnz(0) {}   
    FisherProblem(double* x0_, double* w_, double* u_val_, double* b_,
                  int* col_ind_, int* row_ptr_, double* bounds_,
                  double power_, int row_dim_, int col_dim_, int nnz_)
        : x0(x0_), w(w_), u_val(u_val_), b(b_),
          col_ind(col_ind_), row_ptr(row_ptr_), bounds(bounds_),
          power(power_), row_dim(row_dim_), col_dim(col_dim_), nnz(nnz_) {}
};
struct FisherProblemHost {
    int row_dim, col_dim, nnz;
    double power;
    std::vector<double> x0, w, u_val, b;
    std::vector<int> col_ind, row_ptr;
    // Default constructor initializes empty vectors
    FisherProblemHost()
        : row_dim(0), col_dim(0), nnz(0), power(1.0),
          x0(), w(), u_val(), b(),
          col_ind(), row_ptr() {}
    FisherProblemHost(int row_dim_, int col_dim_, int nnz_, double power_)
        : row_dim(row_dim_), col_dim(col_dim_), nnz(nnz_), power(power_),
          x0(col_dim_, 0.0),
          w(col_dim_, 0.0),
          u_val(nnz_, 0.0),
          b(row_dim_, 0.0),
          col_ind(nnz_, 0),
          row_ptr(row_dim_ + 1, 0)
    {}

    FisherProblemHost(int row_dim_, int col_dim_, int nnz_, double power_,
                      const double* x0_, const double* w_,
                      const double* u_val_, const double* b_,
                      const int* col_ind_, const int* row_ptr_)
        : row_dim(row_dim_), col_dim(col_dim_), nnz(nnz_), power(power_),
          x0(x0_, x0_ + col_dim_),
          w(w_, w_ + col_dim_),
          u_val(u_val_, u_val_ + nnz_),
          b(b_, b_ + row_dim_),
          col_ind(col_ind_, col_ind_ + nnz_),
          row_ptr(row_ptr_, row_ptr_ + row_dim_ + 1)
    {}
};
void print_fisher_problem(const FisherProblem &problem);

void generate_problem_gpu(int row, int col, int nnz, FisherProblem &csr, double power = 1.0, double b_value = 1.0);

FisherProblemHost to_host(const FisherProblem& gpu_problem);
void save_problem_to_files(const FisherProblemHost& prob,
                           const std::string& base_dir,
                           const std::string& stem);