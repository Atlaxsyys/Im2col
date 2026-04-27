#pragma once

#include "tensor.hpp"

#include <cstddef>

enum class GemmBackend
{
    Naive,
    CacheFriendly,
    Intrinsics
};

Tensor conv_im2col (const Tensor& input, const Tensor& kernel);
Tensor conv_im2col (const Tensor& input, const Tensor& kernel, GemmBackend backend);
Tensor conv_im2col (const Tensor& input, const Tensor& kernel,
                    std::size_t stride, std::size_t padding);
Tensor conv_im2col (const Tensor& input, const Tensor& kernel, GemmBackend backend,
                    std::size_t stride, std::size_t padding);
