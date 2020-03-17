cmake_minimum_required(VERSION 3.9)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "--cudart shared")
include_directories(/usr/local/cuda/samples/common/inc)

project(cuda_bimp LANGUAGES CUDA CXX)

# Provide detection::bimp
add_subdirectory(utils)
# Provide detection::utils
add_subdirectory(bimp)

add_subdirectory(example_bimp)
