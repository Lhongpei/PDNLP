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
    bool verbose)
{
    // initialize LBFGSB option
    LBFGSB_CUDA_OPTION<real> lbfgsb_options;

    lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
    lbfgsb_options.mode = LCM_CUDA;
    lbfgsb_options.eps_f = static_cast<real>(1e-50);
    lbfgsb_options.eps_g = static_cast<real>(1e-7);
    lbfgsb_options.eps_x = static_cast<real>(1e-50);
    lbfgsb_options.max_iteration = 1;

    // initialize LBFGSB state
    LBFGSB_CUDA_STATE<real> state;
    memset(&state, 0, sizeof(state));
    real* assist_buffer_cuda = nullptr;
    cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
    if (CUBLAS_STATUS_SUCCESS != stat) {
        std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
        exit(0);
    }
    real minimal_f = std::numeric_limits<real>::max();
    // setup callback function that evaluate function value and its gradient
    state.m_funcgrad_callback = [
        &minimal_f,
        &row,
        &col,
        &nnz,
        &d_u_val,
        &d_w,
        &d_row_ptr,
        &d_col_indice,
        &power,
        &d_p,
        &tau,
        &d_x_old_val,
        &d_utility_no_power,
        &d_obj_tmp,
        &d_x_sum,
        &d_gradient,
        &d_b
    ](
        real* x, real& f, real* g,
        const cudaStream_t& stream,
        const LBFGSB_CUDA_SUMMARY<real>& summary
    ) {

        launch_objective_csr<real>(
            row, col, nnz, x, d_u_val, d_w, d_row_ptr, d_col_indice, power,
            d_obj_tmp, f, d_x_sum, d_p, tau, d_x_old_val, d_b);

        launch_gradient_csr<real>(
            row, x, d_u_val, d_w, d_row_ptr, d_col_indice, power, d_p, tau,
            d_x_old_val, d_utility_no_power, g);
        
        d_gradient = g;
        minimal_f = fmin(minimal_f, f);
        // print_cuarray("solution", x, nnz);
        // printf("Function value: %f, Minimal function value: %f\n", f, minimal_f);
        return 0;
    };
    // initialize number of bounds (0 for this example)
    int *nbd = nullptr;
    real *xl = nullptr;
    real *xu = nullptr;
    cudaMalloc(&xl, nnz * sizeof(real));
    cudaMalloc(&xu, nnz * sizeof(real));
    cudaMalloc(&nbd, nnz * sizeof(int));


    fill(nbd, 1, nnz);
    fill(xl, 1e-6, nnz);
    fill(xu, 1.0 , nnz);
    // cudaMemcpy(d_x_val, xl, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    LBFGSB_CUDA_SUMMARY<real> summary;
    memset(&summary, 0, sizeof(summary));

    auto start_time = std::chrono::steady_clock::now();
    lbfgsbcuda::lbfgsbminimize<real>(
        nnz, state, lbfgsb_options, d_x_val, nbd, xl, xu, summary);
    auto end_time = std::chrono::steady_clock::now();
    auto time_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (verbose) {
        std::cout << "Minimal function value: " << minimal_f << std::endl;
        std::cout << "Number of iterations: " << summary.num_iteration << std::endl;
        std::cout << "Time taken: " << time_duration << " ms; "
                  << "Average time per iteration: "
                  << (time_duration / static_cast<double>(summary.num_iteration))
                  << " ms" << std::endl;
    }


    cudaFree(xl);
    cudaFree(xu);
    cudaFree(nbd);
    cudaFree(assist_buffer_cuda);

    // release cublas
    cublasDestroy(state.m_cublas_handle);
    return summary;
}

template LBFGSB_CUDA_SUMMARY<double> lbfgsb_cuda_primal<double>(
    const int row,
    const int col,
    const int nnz,
    double* d_x_val,
    const double* d_u_val,
    const double* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const double power,
    const double* d_p,
    const double* d_b,
    const double tau,
    const double* d_x_old_val,
    double* d_utility_no_power,
    double* d_gradient,
    double* d_obj_tmp,
    double* d_x_sum,
    bool verbose);

template <typename real>
PdhgLog<real> adaptive_pdhg_fisher(
    FisherProblem &problem,
    PdhgOptions<real> &options
){
    int row_dim = problem.row_dim;
    int col_dim = problem.col_dim;
    int nnz = problem.nnz;

    // Initialize solver state and options
    PdhgLog<real> log;
    PdhgSolverState<real> state;
    ResidualInfo<real> residual_info;
    BufferState<real> buffer;
    RestartInfo<real> restart_info;
    //TODO Delete this line after testing
    // fill(problem.b, 1.0, col_dim);
    cudaMalloc(&buffer.obj_tmp, row_dim * sizeof(real));
    cudaMalloc(&state.current_primal_solution, nnz * sizeof(real));
    cudaMemcpy(state.current_primal_solution, problem.x0, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    // fill(state.current_primal_solution, 1.0, nnz);
    cudaMalloc(&state.current_dual_solution, col_dim * sizeof(real));
    fill(state.current_dual_solution, 1.0, col_dim);
    cudaMalloc(&state.last_primal_solution, nnz * sizeof(real));
    cudaMemcpy(state.last_primal_solution, problem.x0, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMalloc(&state.last_dual_solution, col_dim * sizeof(real));
    cudaMemset(state.last_dual_solution, 0, col_dim * sizeof(real));
    CUDA_CHECK(cudaGetLastError());
    cudaMalloc(&state.avg_primal_solution, nnz * sizeof(real));
    cudaMemcpy(state.avg_primal_solution, problem.x0, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMalloc(&state.avg_dual_solution, col_dim * sizeof(real));
    cudaMemset(state.avg_dual_solution, 0, col_dim * sizeof(real));
    cudaMalloc(&buffer.utility, row_dim * sizeof(real));
    cudaMalloc(&state.primal_gradient, nnz * sizeof(real));
    cudaMalloc(&buffer.current_primal_sum, col_dim * sizeof(real));
    cudaMalloc(&buffer.last_primal_sum, col_dim * sizeof(real));
    cudaMalloc(&buffer.avg_primal_sum, col_dim * sizeof(real));
    csr_column_sum(nnz, col_dim, buffer.current_primal_sum, state.current_primal_solution, problem.col_ind);
    cudaMemcpy(buffer.last_primal_sum, buffer.current_primal_sum, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(buffer.avg_primal_sum, buffer.current_primal_sum, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMalloc(&restart_info.last_restart_primal_solution, nnz * sizeof(real));
    cudaMemcpy(restart_info.last_restart_primal_solution, state.current_primal_solution, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMalloc(&restart_info.last_restart_dual_solution, col_dim * sizeof(real));
    cudaMemcpy(restart_info.last_restart_dual_solution, state.current_dual_solution, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
    real *cache_delta_primal = nullptr;
    real *cache_delta_dual = nullptr;
    real *cache_delta_primal_sum = nullptr;
    cudaMalloc(&cache_delta_primal, nnz * sizeof(real));
    cudaMalloc(&cache_delta_dual, col_dim * sizeof(real));
    cudaMalloc(&cache_delta_primal_sum, col_dim * sizeof(real));
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    state.step_size = 2.0 / sqrt(static_cast<real>(row_dim));
    state.primal_weight = sqrt(static_cast<real>(col_dim)) / static_cast<real>(row_dim);
    bool restart = false;
    double obj;
    CUDA_CHECK(cudaGetLastError());
    //Start Calculation of time
    auto start_time = std::chrono::steady_clock::now();
    while (true){
        // state.outer_solving_time = 0.0f;
        // state.inner_solving_time = 0.0f;
        state.num_outer_iterations += 1;
        restart_info.interval_iterations += 1;
        // if ((state.num_outer_iterations % options.check_frequency == 0) || 
        //     (state.num_outer_iterations == options.max_outer_iterations) ||
        //     (state.numerical_error)) {
        //         break;
        //     }
        for (int i = 0; i < options.max_traceback_times; ++i) {
            double primal_step_size = state.step_size / state.primal_weight;
            double dual_step_size = state.step_size * state.primal_weight;
            cudaMemcpy(state.last_primal_solution, state.current_primal_solution, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(state.last_dual_solution, state.current_dual_solution, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(buffer.last_primal_sum, buffer.current_primal_sum, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            double primal_step_size_inner = primal_step_size;
            LBFGSB_CUDA_SUMMARY<real> summary;
            auto start_subproblem_time = std::chrono::steady_clock::now();
            summary = lbfgsb_cuda_primal(
            row_dim, col_dim, nnz,
            state.current_primal_solution, problem.u_val, problem.w,
            problem.row_ptr, problem.col_ind, problem.power,
            state.current_dual_solution, problem.b, primal_step_size_inner,
            state.last_primal_solution, buffer.utility, state.primal_gradient, buffer.obj_tmp, buffer.current_primal_sum, false);
            auto end_subproblem_time = std::chrono::steady_clock::now();
            log.num_inner_iterations += summary.num_iteration;
            log.inner_solving_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_subproblem_time - start_subproblem_time).count() / 1000.0f;
            launch_utility_csr(row_dim, state.current_primal_solution, problem.u_val, problem.row_ptr, buffer.utility, problem.power);
            dual_update(col_dim, state.current_dual_solution, state.last_dual_solution, buffer.current_primal_sum, buffer.last_primal_sum, problem.b,  dual_step_size);
            if (options.debug) {
                // print_cuarray("current primal solution", state.current_primal_solution, nnz);
                
                print_cuarray("utility", buffer.utility, row_dim);
                
            }

            
            std::tuple<real, real> interaction_movement = compute_interaction_and_movement(
                        col_dim, nnz, state.primal_weight, state.current_primal_solution, state.last_primal_solution, state.current_dual_solution, 
                        state.last_dual_solution, buffer.current_primal_sum, buffer.last_primal_sum, cache_delta_primal, cache_delta_dual, cache_delta_primal_sum, cublas_handle);
                        
            real interaction = std::get<0>(interaction_movement);
            real movement = std::get<1>(interaction_movement);

            real step_size_limit = INFINITY;
            if (interaction > 0.0){
                step_size_limit = movement / interaction;
                if (movement == 0.0) {
                    state.numerical_error = true;
                    break;
                }
            }
            
            real first_term = (1.0 - 1.0 / pow((restart_info.interval_iterations + 1 ), 0.3)) * step_size_limit;
            real second_term = (1.0 + 1.0 / pow((restart_info.interval_iterations + 1), 0.6)) * state.step_size;
            
            state.step_size = fmin(first_term, second_term);

            state.step_size = fmin(fmax(state.step_size, 0.01 / sqrt(static_cast<real>(row_dim) + static_cast<real>(col_dim))), 0.2 / sqrt(static_cast<real>(row_dim) + static_cast<real>(col_dim)));
            if (state.step_size <= step_size_limit) {
                break;
            }
        }



        //=========================Update Average Solutions=========================
        real weight = 1. / (1. +  static_cast<real>(restart_info.interval_iterations));
        
        weighted_self_add_diff(nnz, state.avg_primal_solution, state.current_primal_solution, weight);
        weighted_self_add_diff(col_dim, state.avg_dual_solution, state.current_dual_solution, weight);
        cudaDeviceSynchronize();
        weighted_self_add_diff(col_dim, buffer.avg_primal_sum, buffer.current_primal_sum, weight);

        
        if (state.num_outer_iterations % options.check_frequency == 0) 
        {
        //     //========================Check Residuals=========================
            residual_info.primal_residual = cal_feasibility(col_dim, buffer.current_primal_sum, problem.b, true);
            real gap = 0.0;//cal_gap(nnz, state.current_primal_solution, state.primal_gradient, true);
            residual_info.dual_residual = cal_dual_res(row_dim, col_dim, nnz, problem.w, state.current_primal_solution, problem.u_val, 
                problem.row_ptr, problem.col_ind, state.current_dual_solution, problem.power, true);
            residual_info.residual = fmax(residual_info.primal_residual, residual_info.dual_residual);
            residual_info.avg_primal_residual = cal_feasibility(col_dim, buffer.avg_primal_sum, problem.b, true);
            residual_info.avg_dual_residual = cal_dual_res(row_dim, col_dim, nnz, problem.w, state.avg_primal_solution, problem.u_val, 
                problem.row_ptr, problem.col_ind, state.avg_dual_solution, problem.power, true);
            residual_info.avg_residual = fmax(residual_info.avg_primal_residual, residual_info.avg_dual_residual);
            real better_residual = fmin(residual_info.residual, residual_info.avg_residual);
        
            //========================Check Restart Conditions=========================
            if (better_residual <= 0.2 * restart_info.last_residual) {
                restart = true;
                printf("Restart Condition 1");
            }
            else{
                if (residual_info.residual <= 0.8 * restart_info.last_residual && better_residual >= restart_info.last_residual) {
                    restart = true;
                    printf("Restart Condition 2");
                }
                else{
                    if (restart_info.interval_iterations >= 0.2 * state.num_outer_iterations) {
                        restart = true;
                        printf("Restart Condition 3");
                    }
                    else{
                        restart = false;
                    }
                }
            }

            restart_info.last_residual = better_residual;
            if (restart && state.num_outer_iterations > options.restart_skip_iterations) {
                restart_info.restart_count += 1;
                printf("Current Residual: %f, Average Residual: %f", 
                    residual_info.residual, residual_info.avg_residual);
                printf("Restarting at outer iteration %d, restart count: %d\n", state.num_outer_iterations, restart_info.restart_count);
                if (residual_info.residual < residual_info.avg_residual) {
                    cudaMemcpy(state.avg_dual_solution, state.current_dual_solution, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(state.avg_primal_solution, state.current_primal_solution, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(buffer.avg_primal_sum, buffer.current_primal_sum, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
                    
                    restart_info.last_restart_residual = residual_info.residual;
                }
                else{
                    cudaMemcpy(state.current_primal_solution, state.avg_primal_solution, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(state.current_dual_solution, state.avg_dual_solution, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(buffer.current_primal_sum, buffer.avg_primal_sum, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
                    restart_info.last_restart_residual = residual_info.avg_residual;
                }
                restart_info.interval_iterations = 0;
            }
            
            //========================Update Primal Weight=========================
            restart_info.restart_primal_distant = globalDiffReduce(restart_info.last_restart_primal_solution, state.current_primal_solution, nnz, OP_MAX, true);
            restart_info.restart_dual_distant = globalDiffReduce(restart_info.last_restart_dual_solution, state.current_dual_solution, col_dim, OP_MAX, true);

            cudaMemcpy(restart_info.last_restart_primal_solution, state.current_primal_solution, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(restart_info.last_restart_dual_solution, state.current_dual_solution, col_dim * sizeof(real), cudaMemcpyDeviceToDevice);
            
            state.primal_weight = compute_new_primal_weight(
                restart_info.restart_primal_distant, restart_info.restart_dual_distant, state.primal_weight, options.primal_weight_update_smoothing);

            if (state.primal_weight < 1e-4 || state.primal_weight > 0.5){
                state.primal_weight = sqrt((static_cast<real>(col_dim)/static_cast<real>(row_dim)));
            }
            //========================Print Residuals and Gap=========================
            printf("Outer iteration %d, Inner iteration %d, Primal residual: %f, Dual residual: %f, Gap: %f\n",
                state.num_outer_iterations, log.num_inner_iterations, residual_info.primal_residual, residual_info.dual_residual, gap);
            std::cout << "step size: " << state.step_size << ", primal weight: " << state.primal_weight << std::endl;
        }

        //========================Check Stop Conditions=========================
        if (residual_info.residual < options.tol) {
            printf("Convergence achieved with max gap: %f\n", residual_info.residual);
            break;
        } 
        if (state.num_outer_iterations >= options.max_outer_iterations) {
            printf("Maximum outer iterations reached: %d\n", options.max_outer_iterations);
            break;
       }
    }
    log.num_outer_iterations = state.num_outer_iterations;
    auto end_time = std::chrono::steady_clock::now();
    log.outer_solving_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0f;
    return log;
}

template PdhgLog<double> adaptive_pdhg_fisher<double>(
    FisherProblem &problem,
    PdhgOptions<double> &options
);

