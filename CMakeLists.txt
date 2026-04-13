cmake_minimum_required(VERSION 3.16)
project(im2col_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(im2col_lib
    src/tensor.cpp
    src/matrix.cpp
    src/conv_naive.cpp
)

target_include_directories(im2col_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
