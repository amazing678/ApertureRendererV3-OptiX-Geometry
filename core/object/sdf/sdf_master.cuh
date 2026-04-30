// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "render/object.cuh"
#include "sdf_torus.cuh"
#include "sdf_diamond.cuh"
#include "sdf_volume.cuh"

namespace sdf
{
    extern __device__ SDFVolumeInfo* cachedSDFInfoPtrs;
    extern __device__ uint32_t cachedSDFInfoPtrsCount;
    
    FUNCTION_MODIFIER_INLINE SDFVolumeInfo* GetSDFVolumeInfoPtr(const int cacheIndex)
    {
#if defined(__CUDA_ARCH__)
        if (!cachedSDFInfoPtrs)
        {
            return nullptr;
        }
        if (cacheIndex < 0 || cacheIndex >= cachedSDFInfoPtrsCount)
        {
            return nullptr;
        }
        return &cachedSDFInfoPtrs[cacheIndex];
#else
        return nullptr;
#endif
    }
    
    FUNCTION_MODIFIER_INLINE uint32_t GetSDFVolumeCount()
    {
#if defined(__CUDA_ARCH__)
        return cachedSDFInfoPtrsCount;
#else
        return 0u;
#endif
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectSDFObject(const int depth, const float3 rayOrigin, const float3 rayDirection, const SceneObject& object)
    {
        /*
        {
            const auto& test = cachedSDFInfoPtrs[cachedSDFInfoPtrsCount - 1];
            printf("dim{%d, %d, %d}, ", test.dim_.x, test.dim_.y, test.dim_.z);
            printf("pMax{%.4f, %.4f, %.4f}, ", test.pMax_.x, test.pMax_.y, test.pMax_.z);
            printf("pMin{%.4f, %.4f, %.4f}, ", test.pMin_.x, test.pMin_.y, test.pMin_.z);
            printf("voxelSize{%.4f, %.4f, %.4f}, ", test.voxelSize_.x, test.voxelSize_.y, test.voxelSize_.z);
            printf("unitScale{%.4f}, ", test.unitScale_);
            printf("voxelSizeMin{%.4f}, ", test.voxelSizeMin_);
            printf("volumePtr{%p}", static_cast<void*>(test.volumePtr_));
            printf("\n");
        }
        */
        const ESDFType sdfType = object.additionalObjectInfo_.sdfType_;
        if(sdfType == ESDFType::SDF_TORUS)
        {
            return IntersectSDFObject<SDFTorusFunctor>(depth, rayOrigin, rayDirection, object, SDFTorusFunctor{});
        }
        else if(sdfType == ESDFType::SDF_VOLUME)
        {
            const int volumeIndex = object.additionalObjectInfo_.sdfInfo_.cachedVolumeIndex_;
            if(const SDFVolumeInfo* currentVolumePtr = GetSDFVolumeInfoPtr(volumeIndex))
            {
                const SDFVolumeFunctor volumeFunctor(*currentVolumePtr);
                return IntersectSDFObject<SDFVolumeFunctor, true>(depth, rayOrigin, rayDirection, object, volumeFunctor);
            }
        }
        return {};
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapSDFObject(const float3 rayOrigin, const SceneObject& object)
    {
        const ESDFType sdfType = object.additionalObjectInfo_.sdfType_;
        if(sdfType == ESDFType::SDF_TORUS)
        {
            return OverlapSDFObject<SDFTorusFunctor>(rayOrigin, object, SDFTorusFunctor{});
        }
        else if(sdfType == ESDFType::SDF_VOLUME)
        {
            const int volumeIndex = object.additionalObjectInfo_.sdfInfo_.cachedVolumeIndex_;
            if(const SDFVolumeInfo* currentVolumePtr = GetSDFVolumeInfoPtr(volumeIndex))
            {
                const SDFVolumeFunctor volumeFunctor(*currentVolumePtr);
                return OverlapSDFObject<SDFVolumeFunctor>(rayOrigin, object, volumeFunctor);
            }
        }
        return {};
    }
}
