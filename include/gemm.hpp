#pragma once

#include <cstddef>

void gemm_naive (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n);

void gemm_cache_friendly (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n);

void gemm_intrinsics (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n);
