find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tests
  gemm.cu
)

target_link_libraries(tests PRIVATE
  cl
  GTest::gtest_main
)

add_test(NAME tests COMMAND tests)
