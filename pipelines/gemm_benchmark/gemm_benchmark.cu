#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include "gemm.cuh"
#include "device.cuh"
#include "macros.cuh"
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <map>
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

  float *d_a, *d_b, *d_c;
  nvtxRangePush("matrix memory allocation");
  cudaMalloc(&d_a, n * k * sizeof(float));
  cudaMalloc(&d_b, k * m * sizeof(float));
  cudaMalloc(&d_c, n * m * sizeof(float));
  nvtxRangePop();

  auto F_gemm_tiled = [&]() -> void {
    const int bsz = 32;
    dim3 gdim(n / bsz, m / bsz);
    dim3 bdim(bsz * bsz);
    gemm_tiled<bsz><<<gdim, bdim>>>(d_c, d_a, d_b, n, k, m);
  };

  auto F_gemm_tiled_smem = [&]() -> void {
    const int bsz = 32;
    dim3 gdim(n / bsz, m / bsz);
    dim3 bdim(bsz * bsz);
    gemm_tiled_smem<bsz><<<gdim, bdim>>>(d_c, d_a, d_b, n, k, m);
  };

  cublasHandle_t handle;

  auto F_cublas_sgemm = [&]() -> void {
    float alpha = 1.0, beta = 0.0;
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_b, m, d_a, k, &beta, d_c, m);
  };

  auto F_gemm_1d_blocktiling = [&]() -> void {
    const int BN = 64;
    const int BM = 64;
    const int BK = 8;
    const int TN = 8;
    dim3 gdim(n / BN, m / BM);
    dim3 bdim((BN * BM) / TN);
    gemm_1d_blocktiling<BN, BK, BM, TN><<<gdim, bdim>>>(d_c, d_a, d_b, n, k, m);
  };

  auto kernel_exec = [&](auto &&f, const std::string& name) -> std::pair<std::string,i64> {
    cudaEvent_t tst, tend;
    cudaEventCreate(&tst);
    cudaEventCreate(&tend);
    nvtxRangePush((name + " execution").c_str());
    cudaEventRecord(tst);
    f();
    cudaEventRecord(tend);
    nvtxRangePop();
    cudaDeviceSynchronize();
    return {name, get_microseconds(tst, tend)};
  };

  auto run = [&](int iter_num, std::string name, float eps) -> int {
    cublasCreate_v2(&handle);
    std::map<std::string, i64> times;
    auto rec = [&](const std::pair<std::string, i64> &r) -> void {
      times[r.first] += r.second;
    };
    
    for (int iter=0; iter<iter_num; iter++) {
      rec(kernel_exec(F_gemm_tiled, "gemm_tiled"));

      rec(kernel_exec(F_gemm_tiled_smem, "gemm_tiled_smem"));

      rec(kernel_exec(F_cublas_sgemm, "cublas_sgemm"));

      rec(kernel_exec(F_gemm_1d_blocktiling, "gemm_1d_blocktiling"));
    }

    std::map<std::string, double> flops;
    for (auto &[tk, tv] : times) {
      tv /= iter_num;
      flops[tk] = ((2.0 * n * m * k / tv)) / 1e3;
    }

    for (auto &[tk, tv] : times) {
      std::printf("%s avg %s time: %ld\n", name.c_str(), tk.c_str(), tv);
      std::printf("%s avg %s gflops: %lf\n", name.c_str(), tk.c_str(), flops[tk]);
    }

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
