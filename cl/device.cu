#include "device.cuh"
#include "macros.cuh"
#include <cstdio>

auto init_device(bool list_devices) -> int {
  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));

  if (list_devices) {
    std::printf("Found %d device(s)\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::printf("Device %d: %s\n", i, prop.name);
    }
  }

  CUDA_CALL(cudaSetDevice(0));
  return 0;
}
