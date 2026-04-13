cmake_minimum_required(VERSION 3.16)
project(im2col_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
option(BUILD_BENCHMARKS "Build benchmark executable" ON)

add_library(im2col_lib
    src/tensor.cpp
    src/matrix.cpp
    src/conv_naive.cpp
    src/conv_im2col.cpp
)

target_include_directories(im2col_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_BENCHMARKS)
    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(im2col_bench
        bench/bench_conv.cpp
    )

    target_link_libraries(im2col_bench
        PRIVATE
            im2col_lib
            benchmark::benchmark_main
    )
endif()

if(BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(im2col_tests
        tests/test_main.cpp
    )

    target_link_libraries(im2col_tests
        PRIVATE
            im2col_lib
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(im2col_tests)
endif()
