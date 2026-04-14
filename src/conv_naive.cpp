#include "conv_naive.hpp"

#include <cstddef>

Tensor conv_naive (const Tensor& input, const Tensor& kernel)
{
    const std::size_t n = input.n();
    const std::size_t in_c = input.c();
    const std::size_t in_h = input.h();
    const std::size_t in_w = input.w();

    const std::size_t out_c = kernel.n();
    const std::size_t k_c = kernel.c();
    const std::size_t k_h = kernel.h();
    const std::size_t k_w = kernel.w();

    if (in_c != k_c || in_h < k_h || in_w < k_w)
    {
        return Tensor();
    }

    const std::size_t out_h = in_h - k_h + 1;
    const std::size_t out_w = in_w - k_w + 1;

    Tensor output (n, out_c, out_h, out_w);

    const float* in = input.data();
    const float* k = kernel.data();
    float* out = output.data();

    const std::size_t in_n_stride = in_c * in_h * in_w;
    const std::size_t in_c_stride = in_h * in_w;
    const std::size_t in_h_stride = in_w;

    const std::size_t k_oc_stride = in_c * k_h * k_w;
    const std::size_t k_ic_stride = k_h * k_w;
    const std::size_t k_h_stride = k_w;

    const std::size_t out_n_stride = out_c * out_h * out_w;
    const std::size_t out_c_stride = out_h * out_w;
    const std::size_t out_h_stride = out_w;

    for (std::size_t ni = 0; ni < n; ++ni)
    {
        const float* in_n_base = in + ni * in_n_stride;
        float* out_n_base = out + ni * out_n_stride;

        for (std::size_t oc = 0; oc < out_c; ++oc)
        {
            const float* k_oc_base = k + oc * k_oc_stride;
            float* out_oc_base = out_n_base + oc * out_c_stride;

            for (std::size_t oh = 0; oh < out_h; ++oh)
            {
                float* out_row = out_oc_base + oh * out_h_stride;

                for (std::size_t ow = 0; ow < out_w; ++ow)
                {
                    float sum = 0.0f;

                    for (std::size_t ic = 0; ic < in_c; ++ic)
                    {
                        const float* in_ic_base = in_n_base + ic * in_c_stride;
                        const float* k_ic_base = k_oc_base + ic * k_ic_stride;

                        for (std::size_t kh = 0; kh < k_h; ++kh)
                        {
                            const float* in_row = in_ic_base + (oh + kh) * in_h_stride + ow;
                            const float* k_row = k_ic_base + kh * k_h_stride;

                            for (std::size_t kw = 0; kw < k_w; ++kw)
                            {
                                sum += in_row[kw] * k_row[kw];
                            }
                        }
                    }

                    out_row[ow] = sum;
                }
            }
        }
    }

    return output;
}
