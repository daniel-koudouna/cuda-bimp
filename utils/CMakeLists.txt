cmake_minimum_required(VERSION 3.9)
project(utils)

find_package( OpenCV REQUIRED )

set (CMAKE_CXX_STANDARD 11)
set(SOURCES
        src/utils.cpp
        src/benchmark.cpp
        )

add_library(${PROJECT_NAME} ${SOURCES})
add_library(detection::utils ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC include
  )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})