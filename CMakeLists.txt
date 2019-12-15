cmake_minimum_required(VERSION 3.9)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "--cudart shared")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER "/usr/local/cuda/bin/g++" CACHE FILEPATH "Host compiler to be used by nvcc")
SET(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "")
include_directories(/usr/local/cuda/samples/common/inc)

project(cuda_bimp LANGUAGES CUDA CXX)

# Provide detection::bimp
add_subdirectory(utils)
# Provide detection::utils
add_subdirectory(bimp)

add_subdirectory(example_bimp)
