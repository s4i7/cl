#include "device.cuh"
#include <cstdio>

auto init_device(bool list_devices) -> int {
  int deviceCount;
  auto err = cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::printf("No cuda-compatible devices found.\n");
    return 1;
  }

  if (list_devices) {
    std::printf("Found %d device(s)\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::printf("Device %d: %s\n", i, prop.name);
    }
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    std::printf("Failed to set device: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
