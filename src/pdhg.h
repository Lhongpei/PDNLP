#pragma once
#include "pdhg_struct.h"
#include "lbfgsbcuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "fisher_problem.h"
#include "fisher_func.h"
#include "utils.h"
#include "fisher_dual.h"
#include "increment.h"
#include "reduction.h"
#include "json.hpp"
#include <fstream>
template <typename real>
PdhgLog<real> adaptive_pdhg_fisher(
    FisherProblem &problem,
    PdhgOptions<real> &options
);
template <typename real>
LBFGSB_CUDA_SUMMARY<real> lbfgsb_cuda_primal(
    const int row,
    const int col,
    const int nnz,
    real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const real power,
    const real* d_p,
    const real* d_b,
    const real tau,
    const real* d_x_old_val,
    real* d_utility_no_power,
    real* d_gradient,
    real* d_obj_tmp,
    real* d_x_sum,
    bool verbose);