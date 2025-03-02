#include "gemm.cuh"

__global__ void gemm_naive(float *c, float *a, float *b, int n, int k, int m) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = id / m, col = id % m;
  if (row < n && col < m) {
    float res = 0;
    for (int l=0; l<k; l++) {
      res += a[row * k + l] * b[l * m + col];
    }
    c[row * m + col] = res;
  }
}
