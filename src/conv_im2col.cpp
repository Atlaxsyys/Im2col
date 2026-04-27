#include "conv_im2col.hpp"
#include "gemm.hpp"

#include <cstddef>
#include <vector>

namespace
{
using GemmFn = void (*) (const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

GemmFn select_gemm (GemmBackend backend)
{
    switch (backend)
    {
    case GemmBackend::Naive:
        return &gemm_naive;
    case GemmBackend::CacheFriendly:
        return &gemm_cache_friendly;
    case GemmBackend::Intrinsics:
        return &gemm_intrinsics;
    default:
        return &gemm_naive;
    }
}

void im2col_nchw (
    const float* input_n,
    std::size_t in_c,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t k_h,
    std::size_t k_w,
    std::size_t out_h,
    std::size_t out_w,
    std::size_t stride,
    std::size_t padding,
    float* col)
{
    const std::size_t out_size    = out_h * out_w;
    const std::size_t in_c_stride = in_h * in_w;

    const auto ipad   = static_cast<std::ptrdiff_t>(padding);
    const auto ih_max = static_cast<std::ptrdiff_t>(in_h);
    const auto iw_max = static_cast<std::ptrdiff_t>(in_w);

    std::size_t k_idx = 0;
    for (std::size_t c = 0; c < in_c; ++c)
    {
        const float* input_c = input_n + c * in_c_stride;

        for (std::size_t kh = 0; kh < k_h; ++kh)
        {
            for (std::size_t kw = 0; kw < k_w; ++kw, ++k_idx)
            {
                float*      col_row = col + k_idx * out_size;
                std::size_t out_idx = 0;

                for (std::size_t oh = 0; oh < out_h; ++oh)
                {
                    const auto ih = static_cast<std::ptrdiff_t>(oh * stride + kh) - ipad;

                    for (std::size_t ow = 0; ow < out_w; ++ow, ++out_idx)
                    {
                        const auto iw = static_cast<std::ptrdiff_t>(ow * stride + kw) - ipad;

                        if (ih < 0 || ih >= ih_max || iw < 0 || iw >= iw_max)
                            col_row[out_idx] = 0.0f;
                        else
                            col_row[out_idx] = input_c[ih * in_w + iw];
                    }
                }
            }
        }
    }
}

} // namespace

Tensor conv_im2col (const Tensor& input, const Tensor& kernel,
                    GemmBackend backend,
                    std::size_t stride, std::size_t padding)
{
    const std::size_t batch = input.n();
    const std::size_t in_c  = input.c();
    const std::size_t in_h  = input.h();
    const std::size_t in_w  = input.w();

    const std::size_t out_c = kernel.n();
    const std::size_t k_c   = kernel.c();
    const std::size_t k_h   = kernel.h();
    const std::size_t k_w   = kernel.w();

    const std::size_t padded_h = in_h + 2 * padding;
    const std::size_t padded_w = in_w + 2 * padding;

    if (in_c != k_c || padded_h < k_h || padded_w < k_w)
        return Tensor();

    const std::size_t out_h   = (padded_h - k_h) / stride + 1;
    const std::size_t out_w   = (padded_w - k_w) / stride + 1;
    const std::size_t k_size  = in_c * k_h * k_w;
    const std::size_t out_size = out_h * out_w;

    Tensor output (batch, out_c, out_h, out_w);

    const float* input_data  = input.data();
    const float* kernel_data = kernel.data();
    float*       output_data = output.data();

    const std::size_t in_n_stride  = in_c * in_h * in_w;
    const std::size_t out_n_stride = out_c * out_size;
    const GemmFn      gemm         = select_gemm (backend);

    std::vector<float> col (k_size * out_size);

    for (std::size_t n = 0; n < batch; ++n)
    {
        const float* input_n  = input_data  + n * in_n_stride;
        float*       output_n = output_data + n * out_n_stride;

        im2col_nchw (input_n, in_c, in_h, in_w, k_h, k_w,
                     out_h, out_w, stride, padding, col.data());
        gemm (kernel_data, col.data(), output_n, out_c, k_size, out_size);
    }

    return output;
}

Tensor conv_im2col (const Tensor& input, const Tensor& kernel,
                    std::size_t stride, std::size_t padding)
{
    return conv_im2col (input, kernel, GemmBackend::Naive, stride, padding);
}

Tensor conv_im2col (const Tensor& input, const Tensor& kernel, GemmBackend backend)
{
    return conv_im2col (input, kernel, backend, 1, 0);
}

Tensor conv_im2col (const Tensor& input, const Tensor& kernel)
{
    return conv_im2col (input, kernel, GemmBackend::Naive, 1, 0);
}
