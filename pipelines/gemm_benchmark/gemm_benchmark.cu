#include <algorithm>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "gemm.cuh"
#include "device.cuh"

using i64 = int64_t;
using namespace std;

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

  auto F_gemm_2d_blocktiling = [&]() -> void {
    const int BN = 64;
    const int BM = 64;
    const int BK = 8;
    const int TN = 8;
    const int TM = 8;
    dim3 gdim(n / BN, m / BM);
    dim3 bdim((BN / TN) * (BM / TM));
    gemm_2d_blocktiling<BN, BK, BM, TN, TM><<<gdim, bdim>>>(d_c, d_a, d_b, n, k, m);
  };

  auto kernel_exec = [&](auto &&f, const string& name) -> pair<string,i64> {
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

  auto run = [&](int iter_num, string name, float eps) -> int {
    cublasCreate_v2(&handle);
    map<string, vector<i64>> times;
    auto rec = [&](const pair<string, i64> &r) -> void {
      times[r.first].push_back(r.second);
    };
    auto median = [](vector<i64> v) -> i64 {
      sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    
    for (int iter=0; iter<iter_num; iter++) {
      rec(kernel_exec(F_gemm_tiled, "gemm_tiled"));

      rec(kernel_exec(F_gemm_tiled_smem, "gemm_tiled_smem"));

      rec(kernel_exec(F_cublas_sgemm, "cublas_sgemm"));

      rec(kernel_exec(F_gemm_1d_blocktiling, "gemm_1d_blocktiling"));

      rec(kernel_exec(F_gemm_2d_blocktiling, "gemm_2d_blocktiling"));
    }

    map<string, double> flops;
    map<string, i64> mtm;
    for (auto &[tk, tv] : times) {
      auto med = median(tv);
      mtm[tk] = med;
      flops[tk] = ((2.0 * n * m * k / med)) / 1e3;
    }

    for (auto &[tk, tv] : times) {
      printf("%s p50 %s time: %ld µs\n", name.c_str(), tk.c_str(), mtm[tk]);
      printf("%s p50 %s gflops: %lf\n", name.c_str(), tk.c_str(), flops[tk]);
    }

    cublasDestroy_v2(handle);
    return 0;
  };

  int warmup_runs = 10;
  int perf_runs = 100;

  if(auto err = run(warmup_runs, "warmup", EPS); err != 0) {
    printf("Warmup failed\n");
    return err;
  }
  printf("Warmup OK\n");

  if (auto err = run(perf_runs, "perf", EPS); err != 0) {
    printf("Perf failed\n");
    return err;
  } 
  printf("Perf OK\n");

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
