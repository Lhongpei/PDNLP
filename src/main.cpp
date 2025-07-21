#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "data.h"
#include "fisher_func.h"
#include "lbfgsbcuda.h"
template <typename real>
real lbfgsb_cuda_primal(
    const int m,
    const int n,
    const int nnz,
    real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const int power,
    const real* d_p,
    const real pho,
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
        &m,
        &d_u_val,
        &d_w,
        &d_row_ptr,
        &d_col_indice,
        &power,
        &d_p,
        &pho,
        &d_x_old_val,
        &d_utility_no_power,
        &d_obj_tmp
    ](
        real* x, real& f, real* g,
        const cudaStream_t& stream,
        const LBFGSB_CUDA_SUMMARY<real>& summary
    ) {
        launch_objective_csr<real>(
            m, x, d_u_val, d_w, d_row_ptr, d_col_indice, power, d_obj_tmp,f, d_p, pho,
            d_x_old_val);
        launch_gradient_csr<real>(
            m, x, d_u_val, d_w, d_row_ptr, d_col_indice, power, d_p, pho,
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

int main()
{
    const int m_dim = 100;
    const int n_dim = 100;
    const int nnz = 10000;

    FisherProblem csr;
    generate_problem_gpu(m_dim, n_dim, nnz, csr, 1.0);

    double *h_x0 = new double[nnz];
    cudaMemcpy(h_x0, csr.x0, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "First 5 x0: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_x0[i] << " ";
    std::cout << "\n";

    double *utility = new double[m_dim];
    double *d_utility;
    cudaMalloc(&d_utility, m_dim * sizeof(double));
    double *p;
    cudaMalloc(&p, n_dim * sizeof(double));
    double *tmp_objective;
    cudaMalloc(&tmp_objective, m_dim * sizeof(double));
    launch_utility_csr<double>(
        m_dim, csr.x0, csr.u_val, csr.row_ptr, d_utility, csr.power);
    double obj;
    launch_objective_csr<double>(
        m_dim, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind,
        csr.power, tmp_objective, obj, p, 1.0, csr.x0);
    std::cout << "Objective function launched successfully.\n";
    std::cout << "Objective function value: " << obj << "\n";
    double *h_utility = new double[m_dim];
    cudaMemcpy(h_utility, d_utility, m_dim * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "First 5 utility: ";
    for (int i = 0; i < 100; ++i)
        std::cout << h_utility[i] << " ";
    std::cout << "\n";
    cudaFree(csr.x0);
    cudaFree(csr.w);
    cudaFree(csr.u_val);
    cudaFree(csr.b);
    cudaFree(csr.col_ind);
    cudaFree(csr.row_ptr);
    cudaFree(csr.bounds);
    delete[] h_x0;
    double *gradient;
    cudaMalloc(&gradient, nnz * sizeof(double));
    launch_gradient_csr(
        m_dim, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind, csr.power, p, 1.0, csr.x0, d_utility, gradient);
    double *h_gradient = new double[nnz];
    cudaMemcpy(h_gradient, gradient, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "First 5 gradient values: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_gradient[i] << " ";
    std::cout << "\n";
    double* d_old_x0 = nullptr;
    cudaMemcpy(d_old_x0, csr.x0, nnz * sizeof(double), cudaMemcpyDeviceToDevice);
    lbfgsb_cuda_primal<double>(
        m_dim, n_dim, nnz, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind,
        csr.power, p, 1.0, d_old_x0, d_utility, gradient, tmp_objective);
    // cudaMemset(csr.x0, 3, nnz * sizeof(double));
    launch_gradient_csr(
        m_dim, csr.x0, csr.u_val, csr.w, csr.row_ptr, csr.col_ind, csr.power, p, 1.0, d_old_x0, d_utility, gradient);
        cudaMemcpy(h_gradient, gradient, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "First 5 gradient values: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_gradient[i] << " ";
    std::cout << "\n";
    return 0;
}