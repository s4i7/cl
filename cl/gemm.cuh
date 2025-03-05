#pragma once
#include <cuda_runtime.h>

__global__ void gemm_naive(float *c, float *a, float *b, int n, int k, int m);

template <const int BLOCKSIZE>
__global__ void gemm_tiled(float *c, float *a, float *b, int n, int k, int m) {
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < n && y < m) {
    float tmp = 0.0;
    for (int i = 0; i < k; ++i) {
      tmp += a[x * k + i] * b[i * m + y];
    }
    c[x * m + y] = tmp;
  }
}

template <const int BLOCKSIZE>
__global__ void gemm_tiled_smem(float *c, float *a, float *b, int n, int k, int m) {
  const int crow = blockIdx.x * BLOCKSIZE;
  const int ccol = blockIdx.y * BLOCKSIZE;
  const int threadrow = threadIdx.x / BLOCKSIZE;
  const int threadcol = threadIdx.x % BLOCKSIZE;
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  a += crow * k;
  b += ccol;
  c += crow * m + ccol;

  float tmp = 0.0;

  for (int block_idx = 0; block_idx < k; block_idx += BLOCKSIZE) {
    As[threadrow * BLOCKSIZE + threadcol] = a[threadrow * k + threadcol];
    Bs[threadrow * BLOCKSIZE + threadcol] = b[threadrow * m + threadcol];

    __syncthreads();
    a += BLOCKSIZE;
    b += BLOCKSIZE * m; 

    for (int l=0; l < BLOCKSIZE; l++) {
      tmp += As[threadrow * BLOCKSIZE + l] * Bs[l * BLOCKSIZE + threadcol];
    }

    __syncthreads();
  }

  c[threadrow * m + threadcol] = tmp;
}

template <const int BN, const int BK, const int BM, const int TN>
__global__ void gemm_1d_blocktiling(float *c, float *a, float *b, int n, int k, int m) {
  const int crow = blockIdx.x * BN;
  const int ccol = blockIdx.y * BM;
  const int threadrow = threadIdx.x / BM;
  const int threadcol = threadIdx.x % BM;
  __shared__ float As[BN * BK];
  __shared__ float Bs[BK * BM];

  a += crow * k;
  b += ccol;
  c += crow * m + ccol;

  const int asRow = threadIdx.x / BK;
  const int asCol = threadIdx.x % BK;
  const int bsRow = threadIdx.x / BM;
  const int bsCol = threadIdx.x % BM;

  float tv[TN] = {0.0};

  for (int block_idx = 0; block_idx < k; block_idx += BK) {
    As[asRow * BK + asCol] = a[asRow * k + asCol];
    Bs[bsRow * BM + bsCol] = b[bsRow * m + bsCol];

    __syncthreads();
    a += BK;
    b += BK * m; 

    for (int bsI=0; bsI < BK; bsI++) {
      float bv = Bs[bsI * BM + threadcol];
      for (int rI=0; rI < TN; rI++) {
        tv[rI] += As[(threadrow * TN + rI) * BK + bsI] * bv;
      }
    }

    __syncthreads();
  }

  for (int rI=0; rI < TN; rI++) {
    c[(threadrow * TN + rI) * m + threadcol] = tv[rI];
  }
}
