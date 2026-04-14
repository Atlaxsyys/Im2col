#pragma once

#include "tensor.hpp"

enum class GemmBackend
{
    Naive,
    CacheFriendly,
    Intrinsics
};

Tensor conv_im2col (const Tensor& input, const Tensor& kernel);
Tensor conv_im2col (const Tensor& input, const Tensor& kernel, GemmBackend backend);
