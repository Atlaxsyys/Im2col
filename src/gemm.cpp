#include "gemm.hpp"

#include <algorithm>
#include <cstddef>

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
