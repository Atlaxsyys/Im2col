#include "conv_naive.hpp"

#include <cstddef>

Tensor conv_naive (const Tensor& input, const Tensor& kernel,
                   std::size_t stride, std::size_t padding)
{
    const std::size_t n    = input.n();
    const std::size_t in_c = input.c();
    const std::size_t in_h = input.h();
    const std::size_t in_w = input.w();

    const std::size_t out_c = kernel.n();
    const std::size_t k_c   = kernel.c();
    const std::size_t k_h   = kernel.h();
    const std::size_t k_w   = kernel.w();

    const std::size_t padded_h = in_h + 2 * padding;
    const std::size_t padded_w = in_w + 2 * padding;

    if (in_c != k_c || padded_h < k_h || padded_w < k_w)
        return Tensor();

    const std::size_t out_h = (padded_h - k_h) / stride + 1;
    const std::size_t out_w = (padded_w - k_w) / stride + 1;

    Tensor output (n, out_c, out_h, out_w);

    const float* in  = input.data();
    const float* k   = kernel.data();
    float*       out = output.data();

    const std::size_t in_n_stride  = in_c * in_h * in_w;
    const std::size_t in_c_stride  = in_h * in_w;
    const std::size_t in_h_stride  = in_w;

    const std::size_t k_oc_stride  = in_c * k_h * k_w;
    const std::size_t k_ic_stride  = k_h * k_w;
    const std::size_t k_h_stride   = k_w;

    const std::size_t out_n_stride = out_c * out_h * out_w;
    const std::size_t out_c_stride = out_h * out_w;
    const std::size_t out_h_stride = out_w;

    const auto ipad = static_cast<std::ptrdiff_t>(padding);
    const auto ih_max = static_cast<std::ptrdiff_t>(in_h);
    const auto iw_max = static_cast<std::ptrdiff_t>(in_w);

    for (std::size_t ni = 0; ni < n; ++ni)
    {
        const float* in_n_base  = in  + ni * in_n_stride;
        float*       out_n_base = out + ni * out_n_stride;

        for (std::size_t oc = 0; oc < out_c; ++oc)
        {
            const float* k_oc_base   = k + oc * k_oc_stride;
            float*       out_oc_base = out_n_base + oc * out_c_stride;

            for (std::size_t oh = 0; oh < out_h; ++oh)
            {
                float* out_row = out_oc_base + oh * out_h_stride;

                for (std::size_t ow = 0; ow < out_w; ++ow)
                {
                    float sum = 0.0f;

                    for (std::size_t ic = 0; ic < in_c; ++ic)
                    {
                        const float* in_ic_base = in_n_base + ic * in_c_stride;
                        const float* k_ic_base  = k_oc_base + ic * k_ic_stride;

                        for (std::size_t kh = 0; kh < k_h; ++kh)
                        {
                            const auto ih = static_cast<std::ptrdiff_t>(oh * stride + kh) - ipad;
                            if (ih < 0 || ih >= ih_max)
                                continue;

                            const float* k_row = k_ic_base + kh * k_h_stride;

                            for (std::size_t kw = 0; kw < k_w; ++kw)
                            {
                                const auto iw = static_cast<std::ptrdiff_t>(ow * stride + kw) - ipad;
                                if (iw < 0 || iw >= iw_max)
                                    continue;

                                sum += in_ic_base[ih * in_h_stride + iw] * k_row[kw];
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

Tensor conv_naive (const Tensor& input, const Tensor& kernel)
{
    return conv_naive (input, kernel, 1, 0);
}
