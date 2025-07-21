// #include <iostream>
// #include <cmath>
// #include <vector>
// #include <cuda_runtime.h>

// // 定义目标函数
// __host__ _device__ double objectiveFunction(const double* x, int n) {
//     double y = 0.25 * pow(x[1] - 1, 2);
//     for (int i = 2; i <= n; ++i) {
//         y += pow(x[i] - pow(x[i - 1], 2), 2);
//     }
//     return 4 * y;
// }

// // 定义梯度函数
// __host__ _device__ void gradientFunction(const double* x, double* g, int n) {
//     double t1 = x[2] - pow(x[1], 2);
//     g[1] = 2 * (x[1] - 1) - 1.6e1 * x[1] * t1;
//     for (int i = 2; i < n; ++i) {
//         double t2 = t1;
//         t1 = x[i + 1] - pow(x[i], 2);
//         g[i] = 8 * t2 - 1.6e1 * x[i] * t1;
//     }
//     g[n] = 8 * t1;
// }

// // 定义约束条件
// __host__ _device__ void constraints(const double* x, double* Ax, double* b, int n) {
//     Ax[0] = 0; // 初始化 Ax
//     for (int i = 1; i <= 10; ++i) {
//         Ax[0] += x[i];
//     }
//     Ax[1] = 0;
//     for (int i = 1 + n - 15; i < n; ++i) {
//         Ax[1] += x[i];
//     }
// }

