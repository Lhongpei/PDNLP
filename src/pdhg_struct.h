#pragma once
#include <cuda_runtime.h>
#include <numeric>
template <typename real>
struct PdhgSolverState {
    current_primal_solution: real* = nullptr;
    current_dual_solution: real* = nullptr;
    avg_primal_solution: real* = nullptr;
    avg_dual_solution: real* = nullptr;
    step_size: real = 1.0;
    primal_weight: real = 1.0;
    numerical_error: real = std::numeric_limits<real>::max();
    num_outer_iterations: int = 0;    
    num_inner_iterations: int = 0;
    inner_solving_time: float = 0;
    outer_solving_time: float = 0;
};

template <typename real>
struct ResidualInfo {
    current_primal_residual: real* = nullptr;
    current_dual_residual: real* = nullptr;
    current_gap: real = std::numeric_limits<real>::max();
    avg_primal_residual: real* = nullptr;
    avg_dual_residual: real* = nullptr;
    avg_gap: real = std::numeric_limits<real>::max();
    last_primal_residual: real* = nullptr;
    last_dual_residual: real* = nullptr;
    last_gap: real = std::numeric_limits<real>::max();
};

template <typename real>
struct BufferState {
    utility: real* = nullptr;
    dual_product: real* = nullptr;
    current_primal_sum:real* = nullptr;
    last_primal_sum: real* = nullptr;
    dual_sum: real = std::numeric_limits<real>::max();
};
template <typename real>
struct PdhgOptions {
    restart_skip_iterations: int = 300;
    check_frequency: int = 10;
    max_outer_iterations: int = 10000;
    max_inner_iterations: int = 200;
    primal_weight: real = 1.0;
    step_size: real = 1.0;
}
