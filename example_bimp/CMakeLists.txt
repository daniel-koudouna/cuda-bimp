cmake_minimum_required(VERSION 3.9)

project(example_bimp)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(SOURCES src/example_bimp.cpp)

cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} detection::bimp)
