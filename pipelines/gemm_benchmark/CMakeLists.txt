add_executable(gemm_benchmark
  gemm_benchmark.cu
)

target_link_libraries(gemm_benchmark PRIVATE cl)
