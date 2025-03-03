cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cl LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -fmad=true")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

option(BUILD_TESTS "Build tests" OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl")

add_library(cl)

target_sources(cl
  PRIVATE
    "${SOURCE_DIR}/device.cu"
    "${SOURCE_DIR}/gemm.cu"
    "${SOURCE_DIR}/tensor.cu"
)

find_package(CUDAToolkit REQUIRED)
find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS "/usr/local/cuda/lib64" "/usr/local/cuda/lib" REQUIRED)

target_link_libraries(cl PUBLIC ${NVTOOLSEXT_LIBRARY} CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_include_directories(cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_DIR})

add_subdirectory(pipelines)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


