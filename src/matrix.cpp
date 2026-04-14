#include "matrix.hpp"

#include <algorithm>

Matrix::Matrix()
    : rows_(0), cols_(0), data_()
{
}

Matrix::Matrix (std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols)
{
}

std::size_t Matrix::rows() const noexcept
{
    return rows_;
}

std::size_t Matrix::cols() const noexcept
{
    return cols_;
}

std::size_t Matrix::size() const noexcept
{
    return data_.size();
}

float& Matrix::operator() (std::size_t r, std::size_t c)
{
    return data_[offset(r, c)];
}

const float& Matrix::operator() (std::size_t r, std::size_t c) const
{
    return data_[offset(r, c)];
}

float* Matrix::data() noexcept
{
    return data_.data();
}

const float* Matrix::data() const noexcept
{
    return data_.data();
}

void Matrix::fill (float value) noexcept
{
    std::fill (data_.begin(), data_.end(), value);
}

std::size_t Matrix::offset (std::size_t r, std::size_t c) const noexcept
{
    return r * cols_ + c;
}
