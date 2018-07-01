cmake_minimum_required(VERSION 3.9)

project(bimp_example)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(SOURCES src/example_bimp.cpp)

cuda_add_executable(test_bimp ${SOURCES})
target_link_libraries(test_bimp detection::bimp)
