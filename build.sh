rm -rf build && mkdir build && cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON .. && cmake --build .
