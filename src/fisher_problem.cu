#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "fisher_problem.h"


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
__global__ void fill_rand_b(curandState* state, double* data, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m) {
        data[idx] = curand_uniform_double(&state[idx]) * 25.0; // 生成 [0, 25) 范围内的随机数
    }
}

// 生成 [0, n-1] 范围内的随机整数
__global__ void generate_random_integers(curandState* state, int* col_ind, int nnz, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) {
        col_ind[idx] = static_cast<int>(curand_uniform_double(&state[idx]) * n);
    }
}

// 确保每行的列索引是唯一的
__global__ void unique_columns(int* col_ind, int* row_ptr, int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
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
__global__ void compute_u_sum_dim_1_and_vec_val(double* u_val, int* row_ptr, double* b, double* u_sum_dim_1, double* vec_val, int m) {
    int row = blockIdx.x;
    if (row >= m) return;

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    double sum = 0.0;
    for (int i = start; i < end; ++i) {
        sum += u_val[i];
    }
    u_sum_dim_1[row] = sum;
    vec_val[row] = b[row] / sum; // 避免除零
}

// 计算 x0
__global__ void compute_x0(double* u_val, int* row_ptr, double* vec_val, double* x0, int m) {
    int idx = blockIdx.x;
    if (idx >= m) return;

    int row = idx;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int i = start; i < end; ++i) {
        x0[i] = vec_val[row] * u_val[i];
    }
}

#include <vector>
#include <algorithm>
#include <random>

// m: 行数, n: 列数, nnz: 总非零数
// 返回: row_ptr, col_ind
void build_csr_uniform(int m, int n, int nnz,
                       std::vector<int>& row_ptr,
                       std::vector<int>& col_ind)
{
    row_ptr.assign(m + 1, 0);
    col_ind.reserve(nnz);

    // 1. 每行基础非零个数
    int base = nnz / m;
    int rem  = nnz % m;            // 余数分配给前 rem 行

    // 2. 构造每行计数
    std::vector<int> nnz_per_row(m, base);
    for (int i = 0; i < rem; ++i) nnz_per_row[i] += 1;

    // 3. 随机数引擎
    std::mt19937 gen(std::random_device{}());

    // 4. 逐行生成不重复列索引
    std::vector<int> cols(n);
    for (int i = 0; i < n; ++i) cols[i] = i;

    int ptr = 0;
    for (int i = 0; i < m; ++i) {
        int k = nnz_per_row[i];
        if (k > n) k = n;                 // 防越界
        std::shuffle(cols.begin(), cols.end(), gen);
        col_ind.insert(col_ind.end(), cols.begin(), cols.begin() + k);
        ptr += k;
        row_ptr[i + 1] = ptr;
    }
    col_ind.shrink_to_fit();
}

void generate_problem_gpu(int m, int n, int nnz, FisherProblem &csr, double power) {
    // 1. 设备内存分配（不变）
    cudaMalloc(&csr.x0,      nnz * sizeof(double));
    cudaMalloc(&csr.w,       nnz * sizeof(double));
    cudaMalloc(&csr.u_val,   nnz * sizeof(double));
    cudaMalloc(&csr.b,       m   * sizeof(double));
    cudaMalloc(&csr.col_ind, nnz * sizeof(int));
    cudaMalloc(&csr.row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&csr.bounds,  3 * nnz * sizeof(double));
    double* d_u_sum_dim_1;
    double* d_vec_val;
    cudaMalloc(&d_u_sum_dim_1, m * sizeof(double));
    cudaMalloc(&d_vec_val,     m * sizeof(double));
    csr.power = power;
    csr.m_dim = m;
    csr.n_dim = n;
    csr.nnz = nnz;

    // ---------- 2. CPU 端直接生成合法 CSR ----------
    // 2.1 每行非零个数
    int base = nnz / m;
    int rem  = nnz % m;
    std::vector<int> h_row_ptr(m + 1, 0);
    std::vector<int> h_col_ind;
    h_col_ind.reserve(nnz);

    std::vector<int> nnz_per_row(m, base);
    for (int i = 0; i < rem; ++i) nnz_per_row[i] += 1;

    // 2.2 随机选列（行内不重复）
    std::mt19937 gen(std::random_device{}());
    std::vector<int> cols(n);
    for (int i = 0; i < n; ++i) cols[i] = i;

    int ptr = 0;
    for (int i = 0; i < m; ++i) {
        int k = nnz_per_row[i];
        k = std::min(k, n);

        // 随机采样 k 个不同列
        std::shuffle(cols.begin(), cols.end(), gen);
        std::sort(cols.begin(), cols.begin() + k);   // 按升序排序

        h_col_ind.insert(h_col_ind.end(), cols.begin(), cols.begin() + k);
        ptr += k;
        h_row_ptr[i + 1] = ptr;
    }

    // 2.3 拷贝到 GPU
    cudaMemcpy(csr.row_ptr, h_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr.col_ind, h_col_ind.data(), nnz * sizeof(int),   cudaMemcpyHostToDevice);

    // ---------- 3. 其余随机填充（不变） ----------
    curandState* d_state;
    cudaMalloc(&d_state, nnz * sizeof(curandState));
    setup_rng<<<(nnz + 255) / 256, 256>>>(d_state, 1234ULL, nnz);
    fill_rand<<<(nnz + 255) / 256, 256>>>(d_state, csr.u_val, nnz);
    fill_rand_b<<<(m + 255) / 256, 256>>>(d_state, csr.b, m);

    // ---------- 4. 计算 u_sum_dim_1 和 vec_val ----------
    compute_u_sum_dim_1_and_vec_val<<<m, 1>>>(csr.u_val, csr.row_ptr, csr.b,
                                              d_u_sum_dim_1, d_vec_val, m);
    compute_x0<<<m, 1>>>(csr.u_val, csr.row_ptr, d_vec_val, csr.x0, m);

    // 5. 清理
    cudaFree(d_state);
    cudaFree(d_u_sum_dim_1);
    cudaFree(d_vec_val);
}