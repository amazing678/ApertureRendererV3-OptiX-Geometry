// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>

#include "sdf_master.cuh"

namespace sdf 
{
    __device__ SDFVolumeInfo* cachedSDFInfoPtrs = nullptr;
    __device__ uint32_t cachedSDFInfoPtrsCount = 0;
}