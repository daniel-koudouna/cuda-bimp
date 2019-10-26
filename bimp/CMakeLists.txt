cmake_minimum_required(VERSION 3.9)
project(bimp LANGUAGES CUDA CXX)

set(CMAKE_CUDA_HOST_COMPILER /usr/local/cuda/bin/g++)

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

set (CMAKE_CXX_STANDARD 11)
set(SOURCES
        src/bimp.cpp
        src/convolutionFFT2D.cu
        src/convolutionFFT2D_gold.cpp
        src/cuda_bimp.cpp
        src/cuda_bimp.cu
        )

link_directories(${BOOST_LIBRARY_DIRS})

add_library(${PROJECT_NAME} ${SOURCES})
add_library(detection::bimp ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        )

cuda_add_cufft_to_target(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} detection::utils ${OpenCV_LIBS})
