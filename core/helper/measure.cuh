// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif


#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include "vector.cuh"

inline double DeviceMSE(const float* A, const float* B, const size_t N)
{
    const auto a = thrust::device_pointer_cast(A);
    const auto b = thrust::device_pointer_cast(B);
    const double sum = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(a, b)),
        thrust::make_zip_iterator(thrust::make_tuple(a + N, b + N)),
        [] __device__ (const thrust::tuple<float,float>& t)
        {
            const float d = thrust::get<0>(t) - thrust::get<1>(t);
            return static_cast<double>(d) * d;
        },
        0.0,
        thrust::plus<double>()
    );
    return (N > 0) ? (sum / static_cast<double>(N)) : std::numeric_limits<double>::quiet_NaN();
}

struct LuminanceDiffSqOp
{
    __host__ __device__ double operator()(const thrust::tuple<float3, float3>& t) const
    {
        const float3 a = thrust::get<0>(t);
        const float3 b = thrust::get<1>(t);
        const float La = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
        const float Lb = 0.2126f * b.x + 0.7152f * b.y + 0.0722f * b.z;
        const double d = (double)La - (double)Lb;
        return d * d;
    }
};

inline double DeviceMSELuminance(const float3* A, const float3* B, const size_t N)
{
    if (N == 0 || A == nullptr || B == nullptr)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto first = thrust::make_counting_iterator<size_t>(0);
    auto last = first + N;

    const double sum = thrust::transform_reduce(
        thrust::device,
        first, last,
        [A, B] __host__ __device__(size_t i) 
        {
            const float3 a = A[i];
            const float3 b = B[i];
            const float La = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
            const float Lb = 0.2126f * b.x + 0.7152f * b.y + 0.0722f * b.z;
            const double d = (double)La - (double)Lb;
            return d * d;
        },
        0.0,
        thrust::plus<double>());
    return sum / (double)N;
}