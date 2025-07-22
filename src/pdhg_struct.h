#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <cstddef>

template <typename real>
struct PdhgSolverState {
    real* current_primal_solution = nullptr;
    real* current_dual_solution   = nullptr;
    real* avg_primal_solution     = nullptr;
    real* avg_dual_solution       = nullptr;

    real  step_size         = real{1};
    real  primal_weight     = real{1};
    bool  numerical_error   = false;
    int   num_outer_iterations = 0;
    int   num_inner_iterations = 0;
    float inner_solving_time   = 0.0f;
    float outer_solving_time   = 0.0f;
};

template <typename real>
struct ResidualInfo {
    real* current_primal_residual = nullptr;
    real* current_dual_residual   = nullptr;
    real  current_gap             = std::numeric_limits<real>::max();
    real* avg_primal_residual     = nullptr;
    real* avg_dual_residual       = nullptr;
    real  avg_gap                 = std::numeric_limits<real>::max();
    real* last_primal_residual    = nullptr;
    real* last_dual_residual      = nullptr;
    real  last_gap                = std::numeric_limits<real>::max();
};

template <typename real>
struct BufferState {
    real* utility            = nullptr;
    real* dual_product       = nullptr;
    real* current_primal_sum = nullptr;
    real* last_primal_sum    = nullptr;
    real* avg_primal_sum  = nullptr;
    real  dual_sum           = std::numeric_limits<real>::max();
};

template <typename real>
struct PdhgOptions {
    int  restart_skip_iterations = 300;
    int  check_frequency         = 10;
    int  verbose_frequency       = 100;
    int  max_outer_iterations    = 10'000;
    int  max_inner_iterations    = 200;
    real primal_weight           = real{1};
    real step_size               = real{1};
};

template <typename real>
struct PdhgLog {
    int   num_outer_iterations = 0;
    int   num_inner_iterations = 0;
    real* primal_solution      = nullptr;
    real* dual_solution        = nullptr;
    float outer_solving_time   = 0.0f;
    float inner_solving_time   = 0.0f;
    real* primal_residual      = nullptr;
    real* dual_residual        = nullptr;
};