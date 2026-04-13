#include "conv_naive.hpp"
#include "matrix.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>

TEST(TensorTest, Basics)
{
    Tensor t (2, 3, 4, 5);

    EXPECT_EQ(t.n(), 2u);
    EXPECT_EQ(t.c(), 3u);
    EXPECT_EQ(t.h(), 4u);
    EXPECT_EQ(t.w(), 5u);
    EXPECT_EQ(t.size(), 120u);

    t.fill(3.5f);
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        EXPECT_FLOAT_EQ(t.data()[i], 3.5f);
    }

    t(1, 2, 3, 4) = 9.0f;
    EXPECT_FLOAT_EQ(t(1, 2, 3, 4), 9.0f);

    const std::size_t idx = ((1 * 3 + 2) * 4 + 3) * 5 + 4;
    EXPECT_FLOAT_EQ(t.data()[idx], 9.0f);
}

TEST(MatrixTest, Basics)
{
    Matrix m (3, 4);

    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 4u);
    EXPECT_EQ(m.size(), 12u);

    m.fill(-2.0f);
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        EXPECT_FLOAT_EQ(m.data()[i], -2.0f);
    }

    m(2, 3) = 7.0f;
    EXPECT_FLOAT_EQ(m(2, 3), 7.0f);
    EXPECT_FLOAT_EQ(m.data()[2 * 4 + 3], 7.0f);
}

TEST(ConvNaiveTest, OneByOneKernel)
{
    Tensor input (1, 1, 2, 3);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 0, 2) = 3.0f;
    input(0, 0, 1, 0) = 4.0f;
    input(0, 0, 1, 1) = 5.0f;
    input(0, 0, 1, 2) = 6.0f;

    Tensor kernel (1, 1, 1, 1);
    kernel(0, 0, 0, 0) = 2.0f;

    const Tensor out = conv_naive (input, kernel);
    EXPECT_EQ(out.n(), 1u);
    EXPECT_EQ(out.c(), 1u);
    EXPECT_EQ(out.h(), 2u);
    EXPECT_EQ(out.w(), 3u);

    EXPECT_FLOAT_EQ(out(0, 0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 0, 2), 6.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 1, 0), 8.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 1, 1), 10.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 1, 2), 12.0f);
}

TEST(ConvNaiveTest, MultiChannelMultiOutput)
{
    Tensor input (1, 2, 3, 3);

    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 0, 2) = 3.0f;
    input(0, 0, 1, 0) = 4.0f;
    input(0, 0, 1, 1) = 5.0f;
    input(0, 0, 1, 2) = 6.0f;
    input(0, 0, 2, 0) = 7.0f;
    input(0, 0, 2, 1) = 8.0f;
    input(0, 0, 2, 2) = 9.0f;

    input(0, 1, 0, 0) = 9.0f;
    input(0, 1, 0, 1) = 8.0f;
    input(0, 1, 0, 2) = 7.0f;
    input(0, 1, 1, 0) = 6.0f;
    input(0, 1, 1, 1) = 5.0f;
    input(0, 1, 1, 2) = 4.0f;
    input(0, 1, 2, 0) = 3.0f;
    input(0, 1, 2, 1) = 2.0f;
    input(0, 1, 2, 2) = 1.0f;

    Tensor kernel (2, 2, 2, 2);

    kernel(0, 0, 0, 0) = 1.0f;
    kernel(0, 0, 0, 1) = 0.0f;
    kernel(0, 0, 1, 0) = 0.0f;
    kernel(0, 0, 1, 1) = -1.0f;

    kernel(0, 1, 0, 0) = 0.5f;
    kernel(0, 1, 0, 1) = 0.0f;
    kernel(0, 1, 1, 0) = 0.0f;
    kernel(0, 1, 1, 1) = -0.5f;

    kernel(1, 0, 0, 0) = 1.0f;
    kernel(1, 0, 0, 1) = 1.0f;
    kernel(1, 0, 1, 0) = 1.0f;
    kernel(1, 0, 1, 1) = 1.0f;

    kernel(1, 1, 0, 0) = 0.0f;
    kernel(1, 1, 0, 1) = 0.0f;
    kernel(1, 1, 1, 0) = 0.0f;
    kernel(1, 1, 1, 1) = 0.0f;

    const Tensor out = conv_naive (input, kernel);
    EXPECT_EQ(out.n(), 1u);
    EXPECT_EQ(out.c(), 2u);
    EXPECT_EQ(out.h(), 2u);
    EXPECT_EQ(out.w(), 2u);

    EXPECT_FLOAT_EQ(out(0, 0, 0, 0), -2.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 0, 1), -2.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 1, 0), -2.0f);
    EXPECT_FLOAT_EQ(out(0, 0, 1, 1), -2.0f);

    EXPECT_FLOAT_EQ(out(0, 1, 0, 0), 12.0f);
    EXPECT_FLOAT_EQ(out(0, 1, 0, 1), 16.0f);
    EXPECT_FLOAT_EQ(out(0, 1, 1, 0), 24.0f);
    EXPECT_FLOAT_EQ(out(0, 1, 1, 1), 28.0f);
}
