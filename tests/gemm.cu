#include <cublas_v2.h>
#include <gtest/gtest.h>
#include "device.cuh"
#include "gemm.cuh"
#include "tensor.cuh"

const float eps = 1e-2;

template <typename F>
auto vCublas(F &&f) -> bool {
  init_device();

  int n = 4096;
  int k = 4096;
  int m = 4096;

  
  float *d_a, *d_b, *d_c_ref, *d_c;
  cudaMalloc(&d_a, n * k * sizeof(float));
  cudaMalloc(&d_b, k * m * sizeof(float));
  cudaMalloc(&d_c_ref, n * m * sizeof(float));
  cudaMalloc(&d_c, n * m * sizeof(float));

  float alpha = 1.0, beta = 0.0;
  int BLOCK_SIZE = 1024;
  int NUM_BLOCKS = (n * m + BLOCK_SIZE - 1) / BLOCK_SIZE;

  curandState *s_a, *s_b;
  cudaMalloc(&s_a, n * k * sizeof(curandState));
  cudaMalloc(&s_b, k * m * sizeof(curandState));

  bool *d_res, h_res;
  cudaMalloc(&d_res, sizeof(bool));

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_a, time(nullptr), n, k);
  initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_b, time(nullptr), k, m);

  generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, s_a, n, k);
  generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_b, s_b, k, m);

  cudaDeviceSynchronize();

  f(d_a, d_b, d_c, n, k, m);

  cudaDeviceSynchronize();

  cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_b, m, d_a, k, &beta, d_c_ref, m);

  cudaDeviceSynchronize();

  h_res = true;
  cudaMemcpy(d_res, &h_res, sizeof(bool), cudaMemcpyHostToDevice);
  
  check_matrix_equality_atomic<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_c, d_c_ref, n, m, d_res, eps);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_ref);
  cudaFree(s_a);
  cudaFree(s_b);
  cudaFree(d_res);
  cublasDestroy_v2(handle);
  return h_res;
};

TEST(GEMM_NAIVE_TEST, CheckEqualToCublas) {
  auto f = [&](float *d_a, float *d_b, float *d_c, int n, int k, int m) {
    int BLOCK_SIZE = 1024;
    int NUM_BLOCKS = (n * m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gemm_naive<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_c, d_a, d_b, n, k, m);
  };
  EXPECT_TRUE(vCublas(f));
}

TEST(GEMM_TILED_TEST, CheckEqualToCublas) {
  auto f = [&](float *d_a, float *d_b, float *d_c, int n, int k, int m) -> void {
    dim3 gridDim(4096 / 32, 4096 / 32);
    dim3 blockDim(32 * 32);
    gemm_tiled<32><<<gridDim, blockDim>>>(d_c, d_a, d_b, n, k, m);
  };

  EXPECT_TRUE(vCublas(f));
}

TEST(GEMM_TILED_SMEM_TEST, CheckEqualToCublas) {
  auto f = [&](float *d_a, float *d_b, float *d_c, int n, int k, int m) {
    const int bsz = 32;

    dim3 gridDim(4096 / bsz, 4096 / bsz);
    dim3 blockDim(bsz * bsz);

    gemm_tiled_smem<bsz><<<gridDim, blockDim>>>(d_c, d_a, d_b, n, k, m);
  };

  EXPECT_TRUE(vCublas(f));
}

TEST(GEMM_1D_BLOCKTILING_TEST, CheckEqualToCublas) {
  auto f = [&](float *d_a, float *d_b, float *d_c, int n, int k, int m) {
    const int BN = 64;
    const int BK = 8;
    const int BM = 64;
    const int TN = 8;

    dim3 gd(4096 / BN, 4096 / BM);
    dim3 bd((BN * BM) / TN);

    gemm_1d_blocktiling<BN, BK, BM, TN><<<gd, bd>>>(d_c, d_a, d_b, n, k, m);
  };

  EXPECT_TRUE(vCublas(f));
}

TEST(GEMM_2D_BLOCKTILING_TEST, CheckEqualToCublas) {
  auto f = [&](float *d_a, float *d_b, float *d_c, int n, int k, int m) {
    const int BN = 64;
    const int BK = 8;
    const int BM = 64;
    const int TN = 8;
    const int TM = 8;

    dim3 gd(4096 / BN, 4096 / BM);
    dim3 bd((BN / TN) * (BM / TM));

    gemm_2d_blocktiling<BN, BK, BM, TN, TM><<<gd, bd>>>(d_c, d_a, d_b, n, k, m);
  };

  EXPECT_TRUE(vCublas(f));
}
