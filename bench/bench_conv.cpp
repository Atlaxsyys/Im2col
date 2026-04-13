#include "conv_im2col.hpp"
#include "conv_naive.hpp"
#include "tensor.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>

namespace
{
constexpr std::array<std::array<int64_t, 7>, 3> kCases = {{
    {{1, 3, 32, 32, 16, 3, 3}},
    {{1, 8, 64, 64, 16, 3, 3}},
    {{4, 3, 32, 32, 16, 3, 3}},
}};

void ConvArgs (benchmark::internal::Benchmark* b)
{
    for (const auto& c : kCases)
    {
        b->Args({c[0], c[1], c[2], c[3], c[4], c[5], c[6]});
    }
}

void BM_ConvNaive (benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t in_c = static_cast<std::size_t>(state.range(1));
    const std::size_t in_h = static_cast<std::size_t>(state.range(2));
    const std::size_t in_w = static_cast<std::size_t>(state.range(3));
    const std::size_t out_c = static_cast<std::size_t>(state.range(4));
    const std::size_t k_h = static_cast<std::size_t>(state.range(5));
    const std::size_t k_w = static_cast<std::size_t>(state.range(6));

    Tensor input(n, in_c, in_h, in_w);
    Tensor kernel(out_c, in_c, k_h, k_w);
    input.fill_random(7u, -1.0f, 1.0f);
    kernel.fill_random(700u, -1.0f, 1.0f);

    for (auto _ : state)
    {
        const Tensor out = conv_naive(input, kernel);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

void BM_ConvIm2col (benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t in_c = static_cast<std::size_t>(state.range(1));
    const std::size_t in_h = static_cast<std::size_t>(state.range(2));
    const std::size_t in_w = static_cast<std::size_t>(state.range(3));
    const std::size_t out_c = static_cast<std::size_t>(state.range(4));
    const std::size_t k_h = static_cast<std::size_t>(state.range(5));
    const std::size_t k_w = static_cast<std::size_t>(state.range(6));

    Tensor input(n, in_c, in_h, in_w);
    Tensor kernel(out_c, in_c, k_h, k_w);
    input.fill_random(7u, -1.0f, 1.0f);
    kernel.fill_random(700u, -1.0f, 1.0f);

    for (auto _ : state)
    {
        const Tensor out = conv_im2col(input, kernel);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}
} // namespace

BENCHMARK(BM_ConvNaive)->Apply(ConvArgs);
BENCHMARK(BM_ConvIm2col)->Apply(ConvArgs);
BENCHMARK_MAIN();
