#include "gemm.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <vector>

namespace
{
constexpr std::array<int64_t, 4> kSizes = {64, 128, 256, 512};

void GemmArgs (benchmark::internal::Benchmark* b)
{
    for (int64_t n : kSizes)
        b->Arg(n);
}

void BM_GemmNaive (benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<float> a(n * n), b(n * n), c(n * n);
    for (std::size_t i = 0; i < n * n; ++i) a[i] = static_cast<float>(i % 17) / 17.0f;
    for (std::size_t i = 0; i < n * n; ++i) b[i] = static_cast<float>(i % 19) / 19.0f;

    for (auto _ : state)
    {
        gemm_naive(a.data(), b.data(), c.data(), n, n, n);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
}

void BM_GemmCacheFriendly (benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<float> a(n * n), b(n * n), c(n * n);
    for (std::size_t i = 0; i < n * n; ++i) a[i] = static_cast<float>(i % 17) / 17.0f;
    for (std::size_t i = 0; i < n * n; ++i) b[i] = static_cast<float>(i % 19) / 19.0f;

    for (auto _ : state)
    {
        gemm_cache_friendly(a.data(), b.data(), c.data(), n, n, n);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
}

void BM_GemmIntrinsics (benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<float> a(n * n), b(n * n), c(n * n);
    for (std::size_t i = 0; i < n * n; ++i) a[i] = static_cast<float>(i % 17) / 17.0f;
    for (std::size_t i = 0; i < n * n; ++i) b[i] = static_cast<float>(i % 19) / 19.0f;

    for (auto _ : state)
    {
        gemm_intrinsics(a.data(), b.data(), c.data(), n, n, n);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
}
} // namespace

BENCHMARK(BM_GemmNaive)->Apply(GemmArgs);
BENCHMARK(BM_GemmCacheFriendly)->Apply(GemmArgs);
BENCHMARK(BM_GemmIntrinsics)->Apply(GemmArgs);
BENCHMARK_MAIN();
