#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "fisher_problem.h"
#include "fisher_func.h"
#include "lbfgsbcuda.h"
#include "utils.h"
#include "pdhg.h"
#include "pdhg_struct.h"
int main() {
    int run_times = 1;

    int row_dim = 10000000;
    int col_dim = 400;
    int nnz = static_cast<int>(row_dim * 0.2 * col_dim);
    printf("Row dimension: %d, Column dimension: %d, Non-zero elements: %d\n", row_dim, col_dim, nnz);
    double power = 0.5;
    // print_fisher_problem(problem);
    PdhgOptions<double> options;
    options.max_outer_iterations = 20000;
    options.max_inner_iterations = 100;
    options.check_frequency = 120;
    options.verbose_frequency = 100;
    options.tol =  1e-4;
    options.debug = false;
    CUDA_CHECK(cudaGetLastError());

    //Save solving time and iterations use a json file



    for (int i = 0; i < run_times; ++i) {
        FisherProblem problem;    
        auto start_generate_time = std::chrono::steady_clock::now();
        generate_problem_gpu(row_dim, col_dim ,nnz, problem, power, static_cast<double>(col_dim) * 0.25);
        auto end_generate_time = std::chrono::steady_clock::now();
        printf("Problem generation time: %.2f seconds\n", 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_generate_time - start_generate_time).count() / 1000.0f);
        PdhgLog<double> log = adaptive_pdhg_fisher(problem, options);
        std::string running_log_dir = std::string(FILE_IO_DIR) + "/log" + "/ces_row_" + std::to_string(row_dim) + "_col_" + std::to_string(col_dim) + "_nnz_" + std::to_string(nnz) + "_" + std::to_string(power);
        std::filesystem::create_directories(running_log_dir);
        printf("Problem power: %f\n", problem.power);
        FisherProblemHost problem_host = to_host(problem);
        std::string time_prefix = make_time_prefix();
        save_problem_to_files(problem_host, 
            std::string(FILE_IO_DIR) + "/problem" + "/ces_row_" + std::to_string(row_dim) + "_col_" + std::to_string(col_dim) + "_nnz_" + std::to_string(nnz) + "_" + std::to_string(problem.power) + "/" + time_prefix, 
            "fisher_ces");
        std::string log_file_path = running_log_dir + "/" + time_prefix + ".json";
        std::cout << "Saving log to: " << log_file_path << std::endl;
        nlohmann::json log_json;
        log_json["num_outer_iterations"] = log.num_outer_iterations;
        log_json["num_inner_iterations"] = log.num_inner_iterations;
        log_json["outer_solving_time"]   = log.outer_solving_time;
        log_json["inner_solving_time"]   = log.inner_solving_time;

        std::ofstream ofs(log_file_path);
        if (ofs.is_open()) {
            ofs << std::setprecision(6) << std::fixed << log_json.dump(4);
            ofs.close();
        } else {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
        }
        // Free allocated memory
        cudaFree(problem.x0);
        cudaFree(problem.w);
        cudaFree(problem.u_val);
        cudaFree(problem.b);
        cudaFree(problem.col_ind);
        cudaFree(problem.row_ptr);
        cudaFree(problem.bounds);
        printf("Solving time: %.2f seconds\n", log.outer_solving_time);
        std::cout << "Number of outer iterations: " << log.num_outer_iterations << std::endl;
        std::cout << "Number of inner iterations: " << log.num_inner_iterations << std::endl;
        
    }
    return 0;
}