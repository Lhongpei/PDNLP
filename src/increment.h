#pragma once
template <typename real>
std::tuple<real, real> compute_interaction_and_movement(
    int        col,
    int        nnz,
    real*      d_current_primal_solution,
    real*      d_last_primal_solution,
    real*      d_current_dual_solution,
    real*      d_last_dual_solution,
    real*      d_current_primal_sum,
    real*      d_last_primal_sum,
    cublasHandle_t handle);

template<typename real>
real cal_feasibility(
    int nnz,
    const real* x_sum,
    const real* b,
    bool relative
);

template <typename real>
real cal_gap(
    int nnz,
    const real* primal_solution,
    const real* primal_gradient,
    bool relative
);

template <typename real>
real cal_dual_res(
    const int row_dim,
    const int col_dim,
    const int nnz,
    const real* w,
    const real* x_val,
    const real* u_val,
    const int* row_ptr,
    const int* col_ind,
    const real* p,
    const real power,
    bool relative
);

template <typename real>
real compute_new_primal_weight(
    const real primal_distance,
    const real dual_distance,
    const real primal_weight,
    const real primal_weight_update_smoothing
);