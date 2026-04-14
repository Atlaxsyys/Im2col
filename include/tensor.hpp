#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class Tensor
{
public:
    Tensor();
    Tensor (std::size_t n, std::size_t c, std::size_t h, std::size_t w);

    std::size_t n() const noexcept;
    std::size_t c() const noexcept;
    std::size_t h() const noexcept;
    std::size_t w() const noexcept;
    std::size_t size() const noexcept;

    float& operator() (std::size_t n, std::size_t c, std::size_t h, std::size_t w);
    const float& operator() (std::size_t n, std::size_t c, std::size_t h, std::size_t w) const;

    float* data() noexcept;
    const float* data() const noexcept;

    void fill (float value) noexcept;
    void fill_random (std::uint32_t seed, float lo = -1.0f, float hi = 1.0f);

private:
    std::size_t n_;
    std::size_t c_;
    std::size_t h_;
    std::size_t w_;
    std::vector<float> data_;

    std::size_t offset (std::size_t n, std::size_t c, std::size_t h, std::size_t w) const noexcept;
};
