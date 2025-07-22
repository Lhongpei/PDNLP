#include "pdhg_struct.h"
#include "fisher_problem.h"
#include "utils.h"
#include <cmath>

//Note that we don't allow user to define initial step size and primal weight in current implementation.
template <typename real>
PdhgLog<real> adaptive_pdhg_fisher(
    FisherProblem &problem,
    PdhgOptions<real> &options
){
    int m_dim = problem.m_dim;
    int n_dim = problem.n_dim;
    int nnz = problem.nnz;

    // Initialize solver state and options
    real *b = nullptr;
    PdhgLog<real> log;
    PdhgSolverState<real> state;
    ResidualInfo<real> residual_info;
    BufferState<real> buffer;
    cudaMalloc(&b, m_dim * sizeof(real));
    fill(b, 1.0, m_dim);
    cudaMalloc(&state.current_primal_solution, nnz * sizeof(real));
    cudaMemcpy(state.current_primal_solution, problem.x0, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMalloc(&state.current_dual_solution, m_dim * sizeof(real));
    cudaMalloc(&state.avg_primal_solution, nnz * sizeof(real));
    cudaMalloc(&state.avg_dual_solution, m_dim * sizeof(real));
    cudaMalloc(&buffer.utility, n_dim * sizeof(real));
    cudaMalloc(&buffer.current_primal_sum, n_dim * sizeof(real));
    cudaMalloc(&buffer.last_primal_sum, n_dim * sizeof(real));
    cudaMalloc(&buffer.avg_primal_sum, n_dim * sizeof(real));
    csr_column_sum(nnz, buffer.current_primal_sum, state.current_primal_solution, problem.col_ind);
    cudaMemcpy(buffer.last_primal_sum, buffer.current_primal_sum, n_dim * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(buffer.avg_primal_sum, buffer.current_primal_sum, n_dim * sizeof(real), cudaMemcpyDeviceToDevice);
    state.step_size = 2.0 / sqrt(static_cast<real>(m_dim));
    state.primal_weight = sqrt(static_cast<real>(n_dim)) / static_cast<real>(m_dim);
    while true
        state.outer_solving_time = 0.0f;
        state.inner_solving_time = 0.0f;
        state.num_outer_iterations += 1;
        if ((state.num_outer_iterations % options.check_frequency == 0) || 
            (state.num_outer_iterations == options.max_outer_iterations) ||
            (state.numerical_error)) {
                continue;
            }
    return log;
}