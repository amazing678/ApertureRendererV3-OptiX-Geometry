// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"

#define T_QUANTILE_LUT_SIZE  1024
#define T_QUANTILE_MAX_INDEX 1023

#if defined(__CUDACC__)
extern __device__ __constant__ float T005QuantilesDev[T_QUANTILE_LUT_SIZE];
#endif
extern const float T005QuantilesHost[T_QUANTILE_LUT_SIZE];

namespace quantile
{
    FUNCTION_MODIFIER_INLINE
    float LookupTQuantile005(int nu)
    {
        uint32_t idx = (nu <= 0) ? 0u : static_cast<uint32_t>(nu - 1);
        if (idx > T_QUANTILE_MAX_INDEX) idx = T_QUANTILE_MAX_INDEX;

#if defined(__CUDA_ARCH__)
        return T005QuantilesDev[idx];
#else
        return T005QuantilesHost[idx];
#endif
    }
}