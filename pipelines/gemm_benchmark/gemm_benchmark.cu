#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include "gemm.cuh"
#include "device.cuh"
#include "tensor.cuh"
#include <nvtx3/nvToolsExt.h>
#include <string>
using i64 = int64_t;

constexpr float EPS = 1e-2;

auto get_time() -> i64 {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

auto get_microseconds(cudaEvent_t &start, cudaEvent_t &stop) -> i64 {
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  return ms * 1000;
}

auto matmul_naive_vs_cublas() -> int {
  nvtxRangePush("matmul");
  int n = 4096;
  int k = 4096;
  int m = 4096;

  int BLOCK_SIZE = 1024;
  int NUM_BLOCKS = (n * m + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *d_a, *d_b, *d_c_ref, *d_c;
  nvtxRangePush("matrix memory allocation");
  cudaMalloc(&d_a, n * k * sizeof(float));
  cudaMalloc(&d_b, k * m * sizeof(float));
  cudaMalloc(&d_c_ref, n * m * sizeof(float));
  cudaMalloc(&d_c, n * m * sizeof(float));
  nvtxRangePop();

  auto run = [&](int iter_num, std::string name, float eps) -> int {
    float alpha = 1.0, beta = 0.0;
    i64 total_smem_time = 0, total_cublas_time = 0, total_tiled_time = 0;

    curandState *s_a, *s_b;
    cudaMalloc(&s_a, n * k * sizeof(curandState));
    cudaMalloc(&s_b, k * m * sizeof(curandState));

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    for (int iter=0; iter<iter_num; iter++) {
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_a, time(nullptr), n, k);
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_b, time(nullptr), k, m);

      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, s_a, n, k);
      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_b, s_b, k, m);

      cudaDeviceSynchronize();

      cudaEvent_t tiled_start, tiled_end;
      cudaEventCreate(&tiled_start);
      cudaEventCreate(&tiled_end);
      dim3 gridDim(4096 / 32, 4096 / 32);
      dim3 blockDim(32 * 32);
      nvtxRangePush("gemm_tiled execution");
      cudaEventRecord(tiled_start);
      gemm_tiled<32><<<gridDim, blockDim>>>(d_c, d_a, d_b, n, k, m);
      cudaEventRecord(tiled_end);
      nvtxRangePop();

      cudaDeviceSynchronize();

      cudaEvent_t smem_start, smem_end;
      cudaEventCreate(&smem_start);
      cudaEventCreate(&smem_end);
      const int blocksz = 32;
      dim3 gridDimSmem(n / blocksz, m / blocksz);
      dim3 blockDimSmem(blocksz * blocksz);
      nvtxRangePush("gemm_smem execution");
      cudaEventRecord(smem_start);
      gemm_tiled_smem<blocksz><<<gridDimSmem, blockDimSmem>>>(d_c, d_a, d_b, n, k, m);
      cudaEventRecord(smem_end);
      nvtxRangePop();

      cudaDeviceSynchronize();

      cudaEvent_t cublas_start, cublas_end;
      cudaEventCreate(&cublas_start);
      cudaEventCreate(&cublas_end);

      nvtxRangePush("cublas sgemm execution");
      cudaEventRecord(cublas_start);
      cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_b, m, d_a, k, &beta, d_c_ref, m);
      cudaEventRecord(cublas_end);
      nvtxRangePop();

      cudaDeviceSynchronize();

      total_smem_time += get_microseconds(smem_start, smem_end);
      total_cublas_time += get_microseconds(cublas_start, cublas_end);
      total_tiled_time += get_microseconds(tiled_start, tiled_end);
    }

    i64 average_smem_time = total_smem_time / iter_num;
    i64 average_cublas_time = total_cublas_time / iter_num;
    i64 average_tiled_time = total_tiled_time / iter_num;
    double average_smem_flops = ((2.0 * n * m * k) / average_smem_time) / 1e3;
    double average_coalesced_flops = ((2.0 * n * m * k) / average_tiled_time) / 1e3;

    std::printf("%s avg smem time: %ld\n", name.c_str(), average_smem_time);
    std::printf("%s avg smem gflops: %lf\n", name.c_str(), average_smem_flops);
    std::printf("%s avg tiled time: %ld\n", name.c_str(), average_tiled_time);
    std::printf("%s avg tiled gflops: %lf\n", name.c_str(), average_coalesced_flops);
    std::printf("%s avg cublas time: %ld\n", name.c_str(), average_cublas_time);

    cudaFree(s_a);
    cudaFree(s_b);
    cublasDestroy_v2(handle);
    return 0;
  };

  int warmup_runs = 10;
  int perf_runs = 100;

  if(auto err = run(warmup_runs, "warmup", EPS); err != 0) {
    std::printf("Warmup failed\n");
    return err;
  }
  std::printf("Warmup successfull, naive = cublas\n");

  if (auto err = run(perf_runs, "perf", EPS); err != 0) {
    std::printf("Perf failed\n");
    return err;
  } 
  std::printf("Perf successfull, naive = cublas\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_ref);
  return 0;
}

auto main() -> int {
  if (auto err = init_device(); err != 0) {
    return err;
  }
  if (auto err = matmul_naive_vs_cublas(); err != 0) {
    return err;
  }
  return 0;
}
