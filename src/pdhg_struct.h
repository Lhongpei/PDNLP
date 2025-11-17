#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <cstddef>

template <typename real>
struct PdhgSolverState {
    real* current_primal_solution = nullptr;
    real* current_dual_solution   = nullptr;
    real* last_primal_solution     = nullptr;
    real* last_dual_solution       = nullptr;
    real* avg_primal_solution     = nullptr;
    real* avg_dual_solution       = nullptr;

    real* primal_gradient       = nullptr;

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
    real primal_residual = static_cast<real>(1.0);
    real dual_residual   = static_cast<real>(1.0);
    real best_primal_residual = static_cast<real>(1.0);
    real best_dual_residual   = static_cast<real>(1.0);
    real avg_primal_residual = static_cast<real>(1.0);
    real avg_dual_residual   = static_cast<real>(1.0);
    real residual = static_cast<real>(1.0);
    real avg_residual = static_cast<real>(1.0);
};

template <typename real>
struct RestartInfo {
    real* last_restart_primal_solution = nullptr;
    real* last_restart_dual_solution   = nullptr;
    real last_residual = static_cast<real>(1.0);
    real last_restart_residual = static_cast<real>(1.0);
    int restart_count = 0;
    real restart_primal_distant = static_cast<real>(1.0);
    real restart_dual_distant   = static_cast<real>(1.0);
    int interval_iterations = 0;
};

template <typename real>
struct BufferState {
    real* obj_tmp           = nullptr;
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
    int  max_traceback_times     = 20;
    real primal_weight           = real{1};
    real step_size               = real{1};
    real tol                     = static_cast<real>(1e-6);
    real primal_weight_update_smoothing = static_cast<real>(0.2);
    bool debug                  = false;
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


struct HessianBufferCES {
    double* d = nullptr;
    double* D_inv = nullptr;
    double* D_inv_d = nullptr;
};
