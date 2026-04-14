#include "conv_im2col.hpp"
#include "conv_naive.hpp"
#include "gemm.hpp"
#include "matrix.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

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

TEST(ConvParityTest, Im2colMatchesNaiveOnRandomShapes)
{
    struct Case
    {
        std::size_t n;
        std::size_t in_c;
        std::size_t in_h;
        std::size_t in_w;
        std::size_t out_c;
        std::size_t k_h;
        std::size_t k_w;
        std::uint32_t input_seed;
        std::uint32_t kernel_seed;
    };

    const Case cases[] = {
        {1, 1,  5,  5, 1, 3, 3, 11u, 101u},
        {1, 3,  7,  7, 4, 3, 3, 12u, 102u},
        {2, 3,  8,  6, 5, 2, 2, 13u, 103u},
        {2, 4,  9,  9, 6, 3, 2, 14u, 104u},
        {1, 2,  6, 10, 3, 1, 5, 15u, 105u},
        {2, 8, 16, 16, 8, 5, 5, 16u, 106u},
        {2, 8, 16, 16, 8, 7, 7, 17u, 107u},
    };

    for (std::size_t case_idx = 0; case_idx < std::size(cases); ++case_idx)
    {
        const Case& tc = cases[case_idx];

        Tensor input (tc.n, tc.in_c, tc.in_h, tc.in_w);
        Tensor kernel (tc.out_c, tc.in_c, tc.k_h, tc.k_w);
        input.fill_random(tc.input_seed, -1.0f, 1.0f);
        kernel.fill_random(tc.kernel_seed, -1.0f, 1.0f);

        const Tensor out_naive = conv_naive (input, kernel);
        const Tensor out_im2col = conv_im2col (input, kernel);

        EXPECT_EQ(out_im2col.n(), out_naive.n()) << "case #" << case_idx;
        EXPECT_EQ(out_im2col.c(), out_naive.c()) << "case #" << case_idx;
        EXPECT_EQ(out_im2col.h(), out_naive.h()) << "case #" << case_idx;
        EXPECT_EQ(out_im2col.w(), out_naive.w()) << "case #" << case_idx;
        ASSERT_EQ(out_im2col.size(), out_naive.size()) << "case #" << case_idx;

        for (std::size_t i = 0; i < out_naive.size(); ++i)
        {
            EXPECT_NEAR(out_im2col.data()[i], out_naive.data()[i], 1e-4f)
                << "case #" << case_idx << ", element #" << i;
        }
    }
}

TEST(ConvEdgeCaseTest, ReturnsEmptyWhenInputChannelsDoNotMatchKernelChannels)
{
    Tensor input (2, 3, 8, 8);
    Tensor kernel (4, 2, 3, 3);
    input.fill_random(21u, -1.0f, 1.0f);
    kernel.fill_random(121u, -1.0f, 1.0f);

    const Tensor out_naive = conv_naive (input, kernel);
    const Tensor out_im2col = conv_im2col (input, kernel);

    EXPECT_EQ(out_naive.size(), 0u);
    EXPECT_EQ(out_naive.n(), 0u);
    EXPECT_EQ(out_naive.c(), 0u);
    EXPECT_EQ(out_naive.h(), 0u);
    EXPECT_EQ(out_naive.w(), 0u);

    EXPECT_EQ(out_im2col.size(), 0u);
    EXPECT_EQ(out_im2col.n(), 0u);
    EXPECT_EQ(out_im2col.c(), 0u);
    EXPECT_EQ(out_im2col.h(), 0u);
    EXPECT_EQ(out_im2col.w(), 0u);
}

TEST(ConvEdgeCaseTest, ReturnsEmptyWhenKernelIsLargerThanInput)
{
    Tensor input (1, 2, 4, 5);
    Tensor kernel_h_large (3, 2, 5, 3);
    Tensor kernel_w_large (3, 2, 3, 6);
    input.fill_random(22u, -1.0f, 1.0f);
    kernel_h_large.fill_random(122u, -1.0f, 1.0f);
    kernel_w_large.fill_random(123u, -1.0f, 1.0f);

    const Tensor out_naive_h = conv_naive (input, kernel_h_large);
    const Tensor out_im2col_h = conv_im2col (input, kernel_h_large);
    const Tensor out_naive_w = conv_naive (input, kernel_w_large);
    const Tensor out_im2col_w = conv_im2col (input, kernel_w_large);

    EXPECT_EQ(out_naive_h.size(), 0u);
    EXPECT_EQ(out_im2col_h.size(), 0u);
    EXPECT_EQ(out_naive_w.size(), 0u);
    EXPECT_EQ(out_im2col_w.size(), 0u);
}

TEST(ConvEdgeCaseTest, ExactKernelFitProducesOneByOneOutputAndParity)
{
    Tensor input (2, 3, 4, 5);
    Tensor kernel (4, 3, 4, 5);
    input.fill_random(23u, -1.0f, 1.0f);
    kernel.fill_random(124u, -1.0f, 1.0f);

    const Tensor out_naive = conv_naive (input, kernel);
    const Tensor out_im2col = conv_im2col (input, kernel);

    EXPECT_EQ(out_naive.n(), 2u);
    EXPECT_EQ(out_naive.c(), 4u);
    EXPECT_EQ(out_naive.h(), 1u);
    EXPECT_EQ(out_naive.w(), 1u);

    EXPECT_EQ(out_im2col.n(), out_naive.n());
    EXPECT_EQ(out_im2col.c(), out_naive.c());
    EXPECT_EQ(out_im2col.h(), out_naive.h());
    EXPECT_EQ(out_im2col.w(), out_naive.w());
    ASSERT_EQ(out_im2col.size(), out_naive.size());

    for (std::size_t i = 0; i < out_naive.size(); ++i)
    {
        EXPECT_NEAR(out_im2col.data()[i], out_naive.data()[i], 1e-4f) << "element #" << i;
    }
}

TEST(GemmTest, CacheFriendlyMatchesNaive)
{
    struct Case
    {
        std::size_t m;
        std::size_t k;
        std::size_t n;
    };

    const Case cases[] = {
        {1, 1, 1},
        {3, 5, 7},
        {8, 13, 9},
        {31, 17, 33},
        {64, 64, 64},
    };

    for (std::size_t case_idx = 0; case_idx < std::size(cases); ++case_idx)
    {
        const Case& tc = cases[case_idx];
        std::vector<float> a(tc.m * tc.k);
        std::vector<float> b(tc.k * tc.n);
        std::vector<float> c_ref(tc.m * tc.n);
        std::vector<float> c_opt(tc.m * tc.n);

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 8.0f;
        }
        for (std::size_t i = 0; i < b.size(); ++i)
        {
            b[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) / 9.0f;
        }

        gemm_naive(a.data(), b.data(), c_ref.data(), tc.m, tc.k, tc.n);
        gemm_cache_friendly(a.data(), b.data(), c_opt.data(), tc.m, tc.k, tc.n);

        ASSERT_EQ(c_ref.size(), c_opt.size()) << "case #" << case_idx;
        for (std::size_t i = 0; i < c_ref.size(); ++i)
        {
            EXPECT_NEAR(c_ref[i], c_opt[i], 1e-5f) << "case #" << case_idx << ", element #" << i;
        }
    }
}

TEST(GemmTest, IntrinsicsMatchesNaive)
{
    struct Case
    {
        std::size_t m;
        std::size_t k;
        std::size_t n;
    };

    const Case cases[] = {
        {1, 1, 1},
        {3, 5, 7},
        {8, 13, 9},
        {31, 17, 33},
        {64, 64, 64},
    };

    for (std::size_t case_idx = 0; case_idx < std::size(cases); ++case_idx)
    {
        const Case& tc = cases[case_idx];
        std::vector<float> a(tc.m * tc.k);
        std::vector<float> b(tc.k * tc.n);
        std::vector<float> c_ref(tc.m * tc.n);
        std::vector<float> c_opt(tc.m * tc.n);

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 8.0f;
        }
        for (std::size_t i = 0; i < b.size(); ++i)
        {
            b[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) / 9.0f;
        }

        gemm_naive(a.data(), b.data(), c_ref.data(), tc.m, tc.k, tc.n);
        gemm_intrinsics(a.data(), b.data(), c_opt.data(), tc.m, tc.k, tc.n);

        ASSERT_EQ(c_ref.size(), c_opt.size()) << "case #" << case_idx;
        for (std::size_t i = 0; i < c_ref.size(); ++i)
        {
            EXPECT_NEAR(c_ref[i], c_opt[i], 1e-4f) << "case #" << case_idx << ", element #" << i;
        }
    }
}
