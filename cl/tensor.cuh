#pragma once
#include <curand_kernel.h>

__global__ auto initCurandStates(curandState *states, unsigned long seed, int rows, int cols) -> void;

__global__ auto generateRandomMatrix(float *matrix, curandState *states, int rows, int cols) -> void;

__global__ void check_matrix_equality_atomic(const float* A, const float* B, int n, int m, bool* result, float epsilon);
