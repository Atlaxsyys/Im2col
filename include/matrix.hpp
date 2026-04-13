#pragma once

#include <cstddef>
#include <vector>

class Matrix
{
public:
    Matrix();
    Matrix (std::size_t rows, std::size_t cols);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t size() const noexcept;

    float& operator() (std::size_t r, std::size_t c);
    const float& operator() (std::size_t r, std::size_t c) const;

    float* data() noexcept;
    const float* data() const noexcept;

    void fill (float value) noexcept;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;

    std::size_t offset (std::size_t r, std::size_t c) const noexcept;
};
