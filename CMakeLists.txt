cmake_minimum_required(VERSION 3.9)

project(cuda_bimp)

# Provide detection::bimp
add_subdirectory(utils)
# Provide detection::utils
add_subdirectory(bimp)

add_subdirectory(example_bimp)

