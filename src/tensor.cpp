#include "tensor.hpp"

#include <algorithm>
#include <random>
#include <utility>

Tensor::Tensor()
    : n_(0), c_(0), h_(0), w_(0), data_()
{
}

Tensor::Tensor (std::size_t n, std::size_t c, std::size_t h, std::size_t w)
    : n_(n), c_(c), h_(h), w_(w), data_(n * c * h * w)
{
}

std::size_t Tensor::n() const noexcept
{
    return n_;
}

std::size_t Tensor::c() const noexcept
{
    return c_;
}

std::size_t Tensor::h() const noexcept
{
    return h_;
}

std::size_t Tensor::w() const noexcept
{
    return w_;
}

std::size_t Tensor::size() const noexcept
{
    return data_.size();
}

float& Tensor::operator() (std::size_t n, std::size_t c, std::size_t h, std::size_t w)
{
    return data_[offset (n, c, h, w)];
}

const float& Tensor::operator() (std::size_t n, std::size_t c, std::size_t h, std::size_t w) const
{
    return data_[offset (n, c, h, w)];
}

float* Tensor::data() noexcept
{
    return data_.data();
}

const float* Tensor::data() const noexcept
{
    return data_.data();
}

void Tensor::fill (float value) noexcept
{
    std::fill (data_.begin(), data_.end(), value);
}

void Tensor::fill_random (std::uint32_t seed, float lo, float hi)
{
    if (lo > hi)
    {
        std::swap (lo, hi);
    }

    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> dist (lo, hi);

    for (float& x : data_)
    {
        x = dist (rng);
    }
}

std::size_t Tensor::offset (std::size_t n, std::size_t c, std::size_t h, std::size_t w) const noexcept
{
    return ((n * c_ + c) * h_ + h) * w_ + w;
}
