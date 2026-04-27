#pragma once

#include "tensor.hpp"

#include <cstddef>

Tensor conv_naive (const Tensor& input, const Tensor& kernel);
Tensor conv_naive (const Tensor& input, const Tensor& kernel,
                   std::size_t stride, std::size_t padding);
