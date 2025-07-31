#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "fisher_problem.h"
#include "fisher_func.h"
#include "lbfgsbcuda.h"
#include "utils.h"
#include "increment.h"
template <typename real>
real lbfgsb_cuda_primal(
    const int row,
    const int col,
    const int nnz,
    real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const int power,
    const real* d_p,
    const real tau,
    const real* d_x_old_val,
    real* d_utility_no_power,
    real* d_gradient,
    real* d_obj_tmp)
{
    // initialize LBFGSB option
    LBFGSB_CUDA_OPTION<real> lbfgsb_options;

    lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
    lbfgsb_options.mode = LCM_CUDA;
    lbfgsb_options.eps_f = static_cast<real>(1e-8);
    lbfgsb_options.eps_g = static_cast<real>(1e-8);
    lbfgsb_options.eps_x = static_cast<real>(1e-8);
    lbfgsb_options.max_iteration = 1000;

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
        &d_u_val,
        &d_w,
        &d_row_ptr,
        &d_col_indice,
        &power,
        &d_p,
        &tau,
        &d_x_old_val,
        &d_utility_no_power,
        &d_obj_tmp
    ](
        real* x, real& f, real* g,
        const cudaStream_t& stream,
        const LBFGSB_CUDA_SUMMARY<real>& summary
    ) {
        launch_objective_csr<real>(
            row, x, d_u_val, d_w, d_row_ptr, d_col_indice, power, d_obj_tmp,f, d_p, tau,
            d_x_old_val);
        launch_gradient_csr<real>(
            row, x, d_u_val, d_w, d_row_ptr, d_col_indice, power, d_p, tau,
            d_x_old_val, d_utility_no_power, g);
        if (summary.num_iteration % 1 == 0) {
        std::cout << "CUDA iteration " << summary.num_iteration << " F: " << f
                    << std::endl;
        }
        minimal_f = fmin(minimal_f, f);
        return 0;
    };
    // initialize number of bounds (0 for this example)
    int *nbd = nullptr;
    real *xl = nullptr;
    real *xu = nullptr;
    cudaMalloc(&xl, nnz * sizeof(real));
    cudaMalloc(&xu, nnz * sizeof(real));
    cudaMalloc(&nbd, nnz * sizeof(int));


    cudaMemset(xl, 1e-3, nnz * sizeof(real));
    cudaMemset(xu, 1, nnz * sizeof(real));
    cudaMemset(nbd, 1, nnz * sizeof(int));
    cudaMemcpy(d_x_val, xl, nnz * sizeof(real), cudaMemcpyDeviceToDevice);
    LBFGSB_CUDA_SUMMARY<real> summary;
    memset(&summary, 0, sizeof(summary));

    auto start_time = std::chrono::steady_clock::now();
    lbfgsbcuda::lbfgsbminimize<real>(
        nnz, state, lbfgsb_options, d_x_val, nbd, xl, xu, summary);
    auto end_time = std::chrono::steady_clock::now();
    auto time_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Minimal function value: " << minimal_f << std::endl;
    std::cout << "Number of iterations: " << summary.num_iteration << std::endl;
    std::cout << "Time taken: " << time_duration << " ms; "
              << "Average time per iteration: "
              << (time_duration / static_cast<double>(summary.num_iteration))
              << " ms" << std::endl;
    std::cout << "LBFGSB CUDA minimization completed type: "
              << summary.info << std::endl;

    cudaFree(xl);
    cudaFree(xu);
    cudaFree(nbd);
    cudaFree(assist_buffer_cuda);

    // release cublas
    cublasDestroy(state.m_cublas_handle);
    return 0;
}

// int main()
// {
//     const int row_dim = 3;
//     const int col_dim = 2;
//     const int nnz = 6;
//     const double tau = 1.0;
//     FisherProblem csr;
//     generate_problem_gpu(row_dim, col_dim, nnz, csr, 0.5);
//     fill(csr.x0, 4.0, nnz);
//     fill(csr.u_val, 1.0, nnz);
//     fill(csr.w, 1.0, row_dim);
//     double *d_x_old = nullptr;
//     cudaMalloc(&d_x_old, nnz * sizeof(double));
//     cudaMemset(d_x_old, 0, nnz * sizeof(double));
//     fill(d_x_old, 0.5, nnz);
//     double *h_x0 = new double[nnz];
//     cudaMemcpy(h_x0, csr.x0, nnz * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "First 5 x0: ";
//     for (int i = 0; i < nnz; ++i)
//         std::cout << h_x0[i] << " ";
//     std::cout << "\n";
//     int *h_row_ptr = new int[row_dim + 1];
//     cudaMemcpy(h_row_ptr, csr.row_ptr, (row_dim + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//     std::cout << "Row pointers: ";
//     for (int i = 0; i < row_dim + 1; ++i)
//         std::cout << h_row_ptr[i] << " ";
//     std::cout << "\n";
//     int *h_col_ind = new int[nnz];
//     cudaMemcpy(h_col_ind, csr.col_ind, nnz * sizeof(int), cudaMemcpyDeviceToHost);
//     std::cout << "Column indices: ";
//     for (int i = 0; i < nnz; ++i)
//         std::cout << h_col_ind[i] << " ";
//     std::cout << "\n";
//     double *utility = new double[row_dim];
//     double *d_utility;
//     cudaMalloc(&d_utility, row_dim * sizeof(double));
//     double *p;
//     cudaMalloc(&p, col_dim * sizeof(double));
//     cudaMemset(p, 0, col_dim * sizeof(double));
//     fill(p, -1.0, col_dim);
//     double *tmp_objective;
//     cudaMalloc(&tmp_objective, row_dim * sizeof(double));
//     std::cout<< "power: " << csr.power << "\n";
//     launch_utility_csr<double>(
//         row_dim, csr.x0, csr.u_val, csr.row_ptr, d_utility, csr.power);
//     cudaDeviceSynchronize();
//     double *h_utility = new double[row_dim];
//     cudaMemcpy(h_utility, d_utility, row_dim * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "First 5 utility: ";
//     for (int i = 0; i < row_dim; ++i)
//         std::cout << h_utility[i] << " ";
//     std::cout << "\n";
//     double obj;
//     double *x_sum = nullptr;
//     cudaMalloc(&x_sum, col_dim * sizeof(double));
//     cudaMemset(x_sum, 0, col_dim * sizeof(double));
//     launch_objective_csr<double>(
//         row_dim, col_dim, nnz, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind, csr.power,
//         tmp_objective, obj, x_sum, p, tau, d_x_old
//     );
//     std::cout << "Objective value: " << obj << "\n";
//     cudaMemcpy(h_utility, d_utility, row_dim * sizeof(double), cudaMemcpyDeviceToHost);
//     double *h_x_sum = new double[col_dim];
//     cudaMemcpy(h_x_sum, x_sum, col_dim * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "First 5 x_sum: ";
//     for (int i = 0; i < col_dim; ++i)
//         std::cout << h_x_sum[i] << " ";
//     std::cout << "\n";
//     std::cout << "First 5 utility: ";
//     for (int i = 0; i < row_dim; ++i)
//         std::cout << h_utility[i] << " ";
//     std::cout << "\n";
//     double *d_gradient;
//     cudaMalloc(&d_gradient, nnz * sizeof(double));
//     launch_gradient_csr<double>(
//         row_dim, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind, csr.power,
//         p, tau, d_x_old, d_utility, d_gradient
//     );
//     double *h_gradient = new double[nnz];
//     cudaMemcpy(h_gradient, d_gradient, nnz * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "First 5 gradient: ";
//     for (int i = 0; i < nnz; ++i)
//         std::cout << h_gradient[i] << " ";
//     std::cout << "\n";

//     double *b = nullptr;
//     cudaMalloc(&b, col_dim * sizeof(double));
//     fill(b, 1.0, col_dim);
//     double res_feasibility = cal_feasibility(col_dim, x_sum, b, true);
//     std::cout << "Feasibility residual: " << res_feasibility << "\n";

//     double res_dual = cal_dual_res<double>(row_dim, col_dim, nnz, d_utility, csr.w, csr.x0, csr.u_val, csr.row_ptr, csr.col_ind, p, csr.power, true);
//     std::cout << "Dual residual: " << res_dual << "\n";
    
//     return 0;
// }