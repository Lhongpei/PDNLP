#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "fisher_problem.h"
#include "utils.h"
#include "json.hpp"
#include <fstream>
#include "cnpy.h"
// 初始化随机数生成器
__global__ void setup_rng(curandState* state, unsigned long long seed, int nnz) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// 填充随机数
__global__ void fill_rand(curandState* state, double* data, int nnz) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) {
        data[idx] = curand_uniform_double(&state[idx]) * 0.999 + 1e-6; // 生成 (1e-6, 1) 范围内的随机数
    }
}

// 填充随机数 b
__global__ void fill_rand_b(curandState* state, double* data, int row) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < row) {
        data[idx] = curand_uniform_double(&state[idx]) * 25.0; // 生成 [0, 25) 范围内的随机数
    }
}

// 生成 [0, col-1] 范围内的随机整数
__global__ void generate_random_integers(curandState* state, int* col_ind, int nnz, int col) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) {
        col_ind[idx] = static_cast<int>(curand_uniform_double(&state[idx]) * col);
    }
}
__global__ void fill_rand_int(curandState* state, int* dst, int low, int high, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    dst[idx] = low + (int)(curand_uniform(&state[idx]) * (high - low));
}
// 确保每行的列索引是唯一的
__global__ void unique_columns(int* col_ind, int* row_ptr, int row, int col) {
    int row_idx = blockIdx.x;
    if (row_idx >= row) return;

    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    int num_cols = end - start;

    // 使用一个临时数组存储列索引
    extern __shared__ int shared_cols[];
    int* temp_cols = shared_cols;

    // 将列索引加载到共享内存
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        temp_cols[threadIdx.x + i] = col_ind[start + i];
    }
    __syncthreads();

    // 简单的冒泡排序（适用于小数组）
    for (int i = 0; i < num_cols - 1; ++i) {
        for (int j = 0; j < num_cols - i - 1; ++j) {
            if (temp_cols[j] > temp_cols[j + 1]) {
                int temp = temp_cols[j];
                temp_cols[j] = temp_cols[j + 1];
                temp_cols[j + 1] = temp;
            }
        }
    }
    __syncthreads();

    // 去重
    int unique_count = 0;
    for (int i = 0; i < num_cols; ++i) {
        if (i == 0 || temp_cols[i] != temp_cols[i - 1]) {
            col_ind[start + unique_count] = temp_cols[i];
            unique_count++;
        }
    }
}

// 计算 u_sum_dim_1 和 vec_val
__global__ void compute_u_sum_dim_1_and_vec_val(double* u_val, int* row_ptr, double* b, double* u_sum_dim_1, double* vec_val, int row) {
    int row_idx = blockIdx.x;
    if (row_idx >= row) return;

    int start = row_ptr[row_idx];
    int end = row_ptr[row_idx + 1];
    double sum = 0.0;
    for (int i = start; i < end; ++i) {
        sum += u_val[i];
    }
    u_sum_dim_1[row_idx] = sum;
    vec_val[row_idx] = b[row_idx] / sum; // 避免除零
}

__global__ void compute_x0(
    const double* u_val,
    const int*    col_ind,
    const double* vec_val,   // vec_val[col] 是列 j 的缩放因子
    double*       x0,
    int nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int col = col_ind[idx];     // 第 idx 个非零元所在的列
    x0[idx] = u_val[idx] * vec_val[col];
}

#include <vector>
#include <algorithm>
#include <random>

// row: 行数, col: 列数, nnz: 总非零数
// 返回: row_ptr, col_ind
void build_csr_uniform(int row, int col, int nnz,
                       std::vector<int>& row_ptr,
                       std::vector<int>& col_ind)
{
    row_ptr.assign(row + 1, 0);
    col_ind.reserve(nnz);

    // 1. 每行基础非零个数
    int base = nnz / row;
    int rem  = nnz % row;            // 余数分配给前 rem 行

    // 2. 构造每行计数
    std::vector<int> nnz_per_row(row, base);
    for (int i = 0; i < rem; ++i) nnz_per_row[i] += 1;

    // 3. 随机数引擎
    std::mt19937 gen(std::random_device{}());

    // 4. 逐行生成不重复列索引
    std::vector<int> cols(col);
    for (int i = 0; i < col; ++i) cols[i] = i;

    int ptr = 0;
    for (int i = 0; i < row; ++i) {
        int k = nnz_per_row[i];
        if (k > col) k = col;                 // 防越界
        std::shuffle(cols.begin(), cols.end(), gen);
        col_ind.insert(col_ind.end(), cols.begin(), cols.begin() + k);
        ptr += k;
        row_ptr[i + 1] = ptr;
    }
    col_ind.shrink_to_fit();
}

#include <vector>
#include <numeric>   // For std::iota
#include <random>    // For std::mt19937, std::random_device
#include <algorithm> // For std::shuffle, std::sort, std::swap
#include <iostream>  // For printf/cout

// Assuming FisherProblem struct and other CUDA functions are defined elsewhere
// struct FisherProblem { ... };
// void uniform_rand_fill(...);
// ... etc.

void generate_problem_gpu(int row, int col, int nnz, FisherProblem &csr, double power, double b_value) {
    // 1. 设备内存分配（不变）
    // This section remains the same as your original code.
    cudaMalloc(&csr.x0,      nnz * sizeof(double));
    cudaMalloc(&csr.w,       row * sizeof(double));
    cudaMalloc(&csr.u_val,   nnz * sizeof(double));
    cudaMalloc(&csr.b,       col * sizeof(double));
    cudaMalloc(&csr.col_ind, nnz * sizeof(int));
    cudaMalloc(&csr.row_ptr, (row + 1) * sizeof(int));
    cudaMalloc(&csr.bounds,  3 * nnz * sizeof(double));
    uniform_rand_fill(csr.w, 0.5, 1.0, row);
    double* d_u_sum_dim_1;
    double* d_vec_val;
    cudaMalloc(&d_u_sum_dim_1, row * sizeof(double));
    cudaMalloc(&d_vec_val,     row * sizeof(double));
    csr.power = power;
    csr.row_dim = row;
    csr.col_dim = col;
    csr.nnz = nnz;

    // ---------- 2. CPU 端高效生成合法 CSR (Optimized) ----------
    std::cout << "Starting optimized CPU-side CSR generation..." << std::endl;

    // 2.1 每行非零个数 (Unchanged)
    int base = nnz / row;
    int rem  = nnz % row;
    std::vector<int> h_row_ptr(row + 1, 0);
    h_row_ptr[0] = 0;
    std::vector<int> nnz_per_row(row, base);
    for (int i = 0; i < rem; ++i) {
        nnz_per_row[i] += 1;
    }
    // Cumulatively sum to create the row pointers
    for (int i = 0; i < row; ++i) {
        h_row_ptr[i + 1] = h_row_ptr[i] + nnz_per_row[i];
    }
    std::cout << "Row pointers calculated." << std::endl;
    // Make sure the total nnz matches the final row pointer
    if (h_row_ptr[row] != nnz) {
        // This can happen due to integer division/rounding; adjust the last element
        nnz = h_row_ptr[row]; 
        std::cout << "Warning: nnz adjusted to " << nnz << " to match row distribution." << std::endl;
        csr.nnz = nnz;
    }

    std::vector<int> h_col_ind(nnz);
    
    // 2.2 高效随机选列 (Efficient random column selection)
    std::mt19937 gen(std::random_device{}());

    // --- Key Change 1: Create reusable data structures ---
    // A single vector with all possible column indices (0, 1, ..., col-1)
    // --- Key Change 1: Reusable structures with a position lookup for O(1) finds ---
    std::vector<int> col_candidates(col);
    std::iota(col_candidates.begin(), col_candidates.end(), 0);
    
    // To guarantee each column is used at least once
    std::vector<int> unselected_cols(col);
    std::iota(unselected_cols.begin(), unselected_cols.end(), 0);
    std::shuffle(unselected_cols.begin(), unselected_cols.end(), gen);

    for (int i = 0; i < row; ++i) {
        int k = nnz_per_row[i];
        if (k == 0) continue;

        int num_to_sample = k;
        int sampling_pool_size = col;
        
        // Pointer to the start of the current row's column data in the final array
        auto row_col_start = h_col_ind.begin() + h_row_ptr[i];

        // --- Key Change 2: Handle guaranteed column by swapping it out of the pool ---
        int guaranteed_col = -1;
        if (!unselected_cols.empty()) {
            guaranteed_col = unselected_cols.back();
            unselected_cols.pop_back();

            // Place the guaranteed column
            *row_col_start = guaranteed_col;

            // Find and swap it to the end of the candidate list to exclude it from sampling
            auto it = std::find(col_candidates.begin(), col_candidates.end(), guaranteed_col);
            std::iter_swap(it, col_candidates.begin() + sampling_pool_size - 1);
            
            sampling_pool_size--; // The pool of candidates is now one smaller
            num_to_sample--;      // We need to sample one less column
        }

        // --- Key Change 3: A true O(k) partial Fisher-Yates shuffle ---
        for (int j = 0; j < num_to_sample; ++j) {
            // Pick a random element from the available pool [j, sampling_pool_size - 1]
            std::uniform_int_distribution<int> distrib(j, sampling_pool_size - 1);
            int rand_idx = distrib(gen);
            // Swap it to the front of the sampling section
            std::swap(col_candidates[j], col_candidates[rand_idx]);
        }
        
        // Copy the k (or k-1) sampled elements directly to the final array
        std::copy(col_candidates.begin(), 
                  col_candidates.begin() + num_to_sample, 
                  row_col_start + (guaranteed_col != -1 ? 1 : 0));

        // --- Key Change 4: Sort the final row data in-place ---
        std::sort(row_col_start, row_col_start + k);
        
        // Restore the candidate pool for the next iteration if we modified it
        if (guaranteed_col != -1) {
             auto it = std::find(col_candidates.begin(), col_candidates.end(), guaranteed_col);
             std::iter_swap(it, col_candidates.begin() + sampling_pool_size);
        }
    }

    printf("nnz = %d, h_col_ind.size() = %zu\n", nnz, h_col_ind.size());
    // 2.3 拷贝到 GPU (Unchanged)
    cudaMemcpy(csr.row_ptr, h_row_ptr.data(), (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr.col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "CPU-side generation complete. Starting GPU computations..." << std::endl;

    // ---------- 3. 其余随机填充（不变） ----------
    // This section also remains the same.
    curandState* d_state;
    cudaMalloc(&d_state, nnz * sizeof(curandState));
    setup_rng<<<(nnz + 255) / 256, 256>>>(d_state, 1234ULL, nnz);
    uniform_rand_fill(csr.u_val, 0.5, 1.0, nnz);
    fill(csr.b, b_value, col);
    double *u_sum = nullptr;
    cudaMalloc(&u_sum, col * sizeof(double));
    csr_column_sum(nnz, col, u_sum, csr.u_val, csr.col_ind);
    self_div(col, u_sum, csr.b);
    compute_x0<<<(nnz + 255) / 256, 256>>>(csr.u_val, csr.col_ind, u_sum, csr.x0, nnz);

    // 5. 清理 (Unchanged)
    cudaFree(d_state);
    cudaFree(d_u_sum_dim_1);
    cudaFree(d_vec_val);
    cudaFree(u_sum); // Don't forget to free u_sum
    printf("------Generate Successfully------\n");
}

void print_fisher_problem(const FisherProblem &problem){
    printf("==========FisherProblem==========\n");
    printf("  row_dim: %d, col_dim: %d, nnz: %d, power: %f\n", 
           problem.row_dim, problem.col_dim, problem.nnz, problem.power);
    print_cuarray("x0", problem.x0, problem.nnz);
    print_cuarray("w", problem.w, problem.row_dim);
    print_cuarray("u_val", problem.u_val, problem.nnz);
    print_cuarray("b", problem.b, problem.col_dim);
    print_cuarray("col_ind", problem.col_ind, problem.nnz);
    print_cuarray("row_ptr", problem.row_ptr, problem.row_dim + 1);
    // print_cuarray("bounds", problem.bounds, 3 * problem.nnz);
    printf("==========End of FisherProblem==========\n");
}

FisherProblemHost to_host(const FisherProblem& gpu_problem) {
    FisherProblemHost host;
    host.row_dim = gpu_problem.row_dim;
    host.col_dim = gpu_problem.col_dim;
    host.nnz     = gpu_problem.nnz;
    host.power   = gpu_problem.power;
    host.x0      = copy_from_device(gpu_problem.x0, gpu_problem.nnz);
    host.w       = copy_from_device(gpu_problem.w, gpu_problem.row_dim);
    host.u_val   = copy_from_device(gpu_problem.u_val, gpu_problem.nnz);
    host.b       = copy_from_device(gpu_problem.b, gpu_problem.col_dim);
    host.col_ind = copy_from_device(gpu_problem.col_ind, gpu_problem.nnz);
    host.row_ptr = copy_from_device(gpu_problem.row_ptr, gpu_problem.row_dim + 1);
    return host;
}
void save_problem_to_files(const FisherProblemHost& prob,
                           const std::string& base_dir,
                           const std::string& stem)
{
    namespace fs = std::filesystem;
    fs::path dir(base_dir);
    fs::create_directories(dir); 

    nlohmann::json j;
    j["row_dim"] = prob.row_dim;
    j["col_dim"] = prob.col_dim;
    j["nnz"]      = prob.nnz;
    j["power"]    = prob.power;
    j["x0"]       = stem + "_x0.npy";
    j["w"]        = stem + "_w.npy";
    j["u_val"]    = stem + "_u_val.npy";
    j["b"]        = stem + "_b.npy";
    j["col_ind"]  = stem + "_col_ind.npy";
    j["row_ptr"]  = stem + "_row_ptr.npy";

    std::ofstream((dir / (stem + "_meta.json")).string()) << j.dump(2);

    auto save = [&](const std::string& name, const auto* data, const std::vector<size_t>& shape) {
        cnpy::npy_save((dir / name).string(), data, shape, "w");
    };
    save(j["x0"],       &prob.x0[0],      {static_cast<size_t>(prob.nnz)});
    save(j["w"],        &prob.w[0],       {static_cast<size_t>(prob.row_dim)});
    save(j["u_val"],    &prob.u_val[0],   {static_cast<size_t>(prob.nnz)});
    save(j["b"],        &prob.b[0],       {static_cast<size_t>(prob.col_dim)});
    save(j["col_ind"],  &prob.col_ind[0], {static_cast<size_t>(prob.nnz)});
    save(j["row_ptr"],  &prob.row_ptr[0], {static_cast<size_t>(prob.row_dim + 1)});
}