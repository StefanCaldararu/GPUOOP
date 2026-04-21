cmake_minimum_required(VERSION 3.16)

project(MatrixProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    src/CPUMatrix.cpp
    src/GPUMatrix.cu
    src/main.cpp
)

add_executable(matrix_app ${SOURCES})

set_target_properties(matrix_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)