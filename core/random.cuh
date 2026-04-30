// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "vector.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ inline uint64_t TOTAL_FRAME_INDEX = 0;
__device__ inline uint64_t GLOBAL_SEED = 0x9e3779b97f4a7c15ULL;
FUNCTION_MODIFIER_INLINE uint32_t Hash11(uint32_t a)
{
    a ^= a >> 17;
    a *= 0xed5ad4bbU;
    a ^= a >> 11;
    a *= 0xac4c1b51U;
    a ^= a >> 15;
    a *= 0x31848babU;
    a ^= a >> 14;
    return a;
}

FUNCTION_MODIFIER_INLINE float Hash11f(uint32_t a)
{
    return (Hash11(a) >> 8) * 0x1p-24f;
}

FUNCTION_MODIFIER_DEVICE_INLINE void InitRand(curandState* seed)
{
    constexpr uint64_t MAX_RANDOM_LENGTH = 1ull << 32;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int pixelIndex = threadId + (blockDim.x * blockDim.y) * blockId;
    //curand_init(static_cast<uint64_t>(Hash11(TOTAL_FRAME_INDEX)) | (static_cast<uint64_t>(Hash11(pixelIndex)) << 32), 0, 0, seed);
    curand_init(GLOBAL_SEED, pixelIndex, TOTAL_FRAME_INDEX * MAX_RANDOM_LENGTH, seed);
}

FUNCTION_MODIFIER_DEVICE_INLINE float Rand1(curandState* seed)
{
    return curand_uniform(seed);
}

FUNCTION_MODIFIER_DEVICE_INLINE float2 Rand2(curandState* seed)
{
    return {curand_uniform(seed), curand_uniform(seed)};
}

FUNCTION_MODIFIER_DEVICE_INLINE float3 Rand3(curandState* seed)
{
    return {curand_uniform(seed), curand_uniform(seed), curand_uniform(seed)};
}