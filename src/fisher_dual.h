#pragma once
#include <cuda_runtime.h>
template <typename real>
void dual_update(
    const int col,
    real* dual_solution,
    const real* last_dual_solution,
    const real* primal_sum,
    const real* last_primal_sum,
    const real* right_hand_side,
    const real sigma
);
