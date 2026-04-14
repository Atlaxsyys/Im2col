#include "gemm.hpp"

#include <algorithm>
#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void gemm_naive (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n)
{
    std::fill(c, c + m * n, 0.0f);

    for (std::size_t i = 0; i < m; ++i)
    {
        const float* a_row = a + i * k;
        float* c_row = c + i * n;

        for (std::size_t p = 0; p < k; ++p)
        {
            const float a_ip = a_row[p];
            if (a_ip == 0.0f)
            {
                continue;
            }

            const float* b_row = b + p * n;
            for (std::size_t j = 0; j < n; ++j)
            {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

void gemm_cache_friendly (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n)
{
    std::fill(c, c + m * n, 0.0f);

    constexpr std::size_t kTileM = 64;
    constexpr std::size_t kTileK = 64;
    constexpr std::size_t kTileN = 64;

    for (std::size_t i0 = 0; i0 < m; i0 += kTileM)
    {
        const std::size_t i_max = std::min(i0 + kTileM, m);

        for (std::size_t p0 = 0; p0 < k; p0 += kTileK)
        {
            const std::size_t p_max = std::min(p0 + kTileK, k);

            for (std::size_t j0 = 0; j0 < n; j0 += kTileN)
            {
                const std::size_t j_max = std::min(j0 + kTileN, n);

                for (std::size_t i = i0; i < i_max; ++i)
                {
                    const float* a_row = a + i * k;
                    float* c_row = c + i * n;

                    for (std::size_t p = p0; p < p_max; ++p)
                    {
                        const float a_ip = a_row[p];
                        if (a_ip == 0.0f)
                        {
                            continue;
                        }

                        const float* b_row = b + p * n;

                        std::size_t j = j0;
                        for (; j + 7 < j_max; j += 8)
                        {
                            c_row[j + 0] += a_ip * b_row[j + 0];
                            c_row[j + 1] += a_ip * b_row[j + 1];
                            c_row[j + 2] += a_ip * b_row[j + 2];
                            c_row[j + 3] += a_ip * b_row[j + 3];
                            c_row[j + 4] += a_ip * b_row[j + 4];
                            c_row[j + 5] += a_ip * b_row[j + 5];
                            c_row[j + 6] += a_ip * b_row[j + 6];
                            c_row[j + 7] += a_ip * b_row[j + 7];
                        }

                        for (; j < j_max; ++j)
                        {
                            c_row[j] += a_ip * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_intrinsics (
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k,
    std::size_t n)
{
    std::fill(c, c + m * n, 0.0f);

    for (std::size_t i = 0; i < m; ++i)
    {
        const float* a_row = a + i * k;
        float* c_row = c + i * n;

        for (std::size_t p = 0; p < k; ++p)
        {
            const float a_ip = a_row[p];
            if (a_ip == 0.0f)
            {
                continue;
            }

            const float* b_row = b + p * n;
            std::size_t j = 0;

#if defined(__AVX2__) && defined(__FMA__)
            const __m256 a_vec = _mm256_set1_ps(a_ip);
            for (; j + 7 < n; j += 8)
            {
                const __m256 c_vec = _mm256_loadu_ps(c_row + j);
                const __m256 b_vec = _mm256_loadu_ps(b_row + j);
                const __m256 out = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(c_row + j, out);
            }
#elif defined(__ARM_NEON)
            const float32x4_t a_vec = vdupq_n_f32(a_ip);
            for (; j + 3 < n; j += 4)
            {
                float32x4_t c_vec = vld1q_f32(c_row + j);
                const float32x4_t b_vec = vld1q_f32(b_row + j);
                c_vec = vmlaq_f32(c_vec, b_vec, a_vec);
                vst1q_f32(c_row + j, c_vec);
            }
#endif
            for (; j < n; ++j)
            {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}
