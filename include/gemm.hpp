#pragma once

#include <cstddef>

void gemm_naive (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m,
    std::size_t k,
    std::size_t n);

void gemm_cache_friendly (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m,
    std::size_t k,
    std::size_t n);

void gemm_intrinsics (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m,
    std::size_t k,
    std::size_t n);
