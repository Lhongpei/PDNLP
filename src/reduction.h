#pragma once
enum ReductionOp { OP_SUM, OP_MIN, OP_MAX };
enum NormType { NORM_L1, NORM_L2, NORM_INF };
template <typename real>
real globalDiffReduce(const real* d_1, const real* d_2, const int N, ReductionOp op, bool abs);

template<typename real>
real globalReduce(const real* d_input, const int N, ReductionOp op);


template<typename real>
real cuNorm(const real* d_input, const int N, NormType norm_type);