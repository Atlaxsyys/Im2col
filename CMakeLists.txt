cmake_minimum_required(VERSION 3.16)
project(im2col_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(im2col_lib
    src/tensor.cpp
    src/matrix.cpp
    src/conv_naive.cpp
)

target_include_directories(im2col_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
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
