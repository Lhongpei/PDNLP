#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "fisher_problem.h"
#include "utils.h"

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

void generate_problem_gpu(int row, int col, int nnz, FisherProblem &csr, double power, double b_value) {
    // 1. 设备内存分配（不变）
    cudaMalloc(&csr.x0,      nnz * sizeof(double));
    cudaMalloc(&csr.w,       row * sizeof(double));
    cudaMalloc(&csr.u_val,   nnz * sizeof(double));
    cudaMalloc(&csr.b,       col   * sizeof(double));
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

    // ---------- 2. CPU 端直接生成合法 CSR ----------
    // 2.1 每行非零个数
    int base = nnz / row;
    int rem  = nnz % row;
    std::vector<int> h_row_ptr(row + 1, 0);
    std::vector<int> h_col_ind;
    h_col_ind.reserve(nnz);

    std::vector<int> nnz_per_row(row, base);
    for (int i = 0; i < rem; ++i) nnz_per_row[i] += 1;

    // 2.2 随机选列（行内不重复）
    std::mt19937 gen(std::random_device{}());
    std::vector<int> cols(col);
    for (int i = 0; i < col; ++i) cols[i] = i;
    // First guaranttee that each col is selected at least once
    std::vector<int> unselected_cols(col);
std::iota(unselected_cols.begin(), unselected_cols.end(), 0);
std::shuffle(unselected_cols.begin(), unselected_cols.end(), gen);

int ptr = 0;
for (int i = 0; i < row; ++i) {
    int k = nnz_per_row[i];
    k = std::min(k, col);

    // 保证每列至少出现一次
    std::vector<int> selected;
    if (!unselected_cols.empty()) {
        selected.push_back(unselected_cols.back());
        unselected_cols.pop_back();
    }

    // 补足剩余 k-1 个（避免重复）
    std::vector<int> available;
    for (int j = 0; j < col; ++j) {
        if (std::find(selected.begin(), selected.end(), j) == selected.end()) {
            available.push_back(j);
        }
    }
    std::shuffle(available.begin(), available.end(), gen);
    int need = k - selected.size();
    selected.insert(selected.end(), available.begin(), available.begin() + need);

    std::sort(selected.begin(), selected.end());
    h_col_ind.insert(h_col_ind.end(), selected.begin(), selected.end());
    ptr += selected.size();
    h_row_ptr[i + 1] = ptr;
}
    printf("nnz = %d, h_col_ind.size() = %zu\n", nnz, h_col_ind.size());

    // 2.3 拷贝到 GPU
    cudaMemcpy(csr.row_ptr, h_row_ptr.data(), (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr.col_ind, h_col_ind.data(), nnz * sizeof(int),   cudaMemcpyHostToDevice);

    // ---------- 3. 其余随机填充（不变） ----------
    curandState* d_state;
    cudaMalloc(&d_state, nnz * sizeof(curandState));
    setup_rng<<<(nnz + 255) / 256, 256>>>(d_state, 1234ULL, nnz);
    uniform_rand_fill(csr.u_val, 0.5, 1.0, nnz);
    // fill(csr.u_val, 1.0, nnz);
    // fill(csr.b, 0.25 * row, col)
    fill(csr.b, b_value, col);
    double *u_sum = nullptr;
    cudaMalloc(&u_sum, col * sizeof(double));
    csr_column_sum(nnz, col, u_sum, csr.u_val, csr.col_ind);
    print_cuarray("u_sum", u_sum, col);
    self_div(col, u_sum, csr.b);
    compute_x0<<<(nnz + 255) / 256, 256>>>(csr.u_val, csr.col_ind, u_sum, csr.x0, nnz);

    // 5. 清理
    cudaFree(d_state);
    cudaFree(d_u_sum_dim_1);
    cudaFree(d_vec_val);
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
