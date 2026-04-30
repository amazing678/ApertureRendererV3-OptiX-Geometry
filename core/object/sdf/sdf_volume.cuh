// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "object/aabb.cuh"

namespace sdf
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE AdditionalObjectInfo CreateSDFVolume(const int volumeIndex)
    {
        return {ESDFType::SDF_VOLUME, SDFInfo{{0.0f, 0.0f, 0.0f, 0.0f}, volumeIndex}};
    }
    struct SDFVolumeInfo
    {
        float* __restrict__ volumePtr_ = nullptr;
        int3 dim_ = {0, 0, 0};
        //float3 pMin_ = {0.0f, 0.0f, 0.0f};
        //float3 pMax_ = {0.0f, 0.0f, 0.0f};
        //float3 voxelSize_ = {0.0f, 0.0f, 0.0f};
        //float voxelSizeWorldMin_ = 0.0f;
        float voxelSizeMax_ = 0.0f;
        //float unitScale_ = 0.0f;
        
        FUNCTION_MODIFIER_INLINE void ComputeVoxelSize() 
        {
            const float3 voxelResInv = {
                1.0f / fmaxf(static_cast<float>(dim_.x), 1.0f),
                1.0f / fmaxf(static_cast<float>(dim_.y), 1.0f),
                1.0f / fmaxf(static_cast<float>(dim_.z), 1.0f)
            };
            //const float3 volumeSize = pMax_ - pMin_;
            /*
            voxelSize_ = make_float3(
                (dim_.x>1) ? volumeSize.x / static_cast<float>(dim_.x - 1) : 0.0f,
                (dim_.y>1) ? volumeSize.y / static_cast<float>(dim_.y - 1) : 0.0f,
                (dim_.z>1) ? volumeSize.z / static_cast<float>(dim_.z - 1) : 0.0f
            );
            */
            
            //voxelSizeWorldMin_ = fminf(voxelSize_.x, fminf(voxelSize_.y, voxelSize_.z));
            //voxelSizeMax_ = fmaxf(voxelResInv.x, fmaxf(voxelResInv.y, voxelResInv.z)) * 1.8f;
            voxelSizeMax_ = length(voxelResInv) * SDF_VOLUME_USE_NEAREST_SCALE;
            //const float3 length = make_float3(volumeSize.x, volumeSize.y, volumeSize.z);
            //unitScale_ = fmaxf(1.0e-20f, fminf(length.x, fminf(length.y, length.z)));
        }
    };
    
    FUNCTION_MODIFIER_INLINE size_t Index3D(const int x, const int y, const int z, const int3 dim) 
    {
        return static_cast<size_t>(z) * dim.x * dim.y + static_cast<size_t>(y) * dim.x + x;
    }
    /*
    FUNCTION_MODIFIER_INLINE float3 ToGridWorld(const float3 posW, const SDFVolumeInfo& info) 
    {
        const float3 g = make_float3(
            (info.voxelSize_.x>0) ? (posW.x - info.pMin_.x) / info.voxelSize_.x : 0.0f,
            (info.voxelSize_.y>0) ? (posW.y - info.pMin_.y) / info.voxelSize_.y : 0.0f,
            (info.voxelSize_.z>0) ? (posW.z - info.pMin_.z) / info.voxelSize_.z : 0.0f
        );
        return g; // in [0..nx-1], [0..ny-1], [0..nz-1]
    }
    */
    FUNCTION_MODIFIER_INLINE float3 ToGrid01(float3 pos01, const SDFVolumeInfo& info)
    {
        // map [0,1] to [0, nx-1], etc.
        return make_float3(
            pos01.x * static_cast<float>(info.dim_.x - 1),
            pos01.y * static_cast<float>(info.dim_.y - 1),
            pos01.z * static_cast<float>(info.dim_.z - 1)
        );
    }
    
    FUNCTION_MODIFIER_INLINE float SampleSDFNearest01(const float3 pos01, const SDFVolumeInfo& info)
    {
        if (!info.volumePtr_ || info.dim_.x * info.dim_.y * info.dim_.z == 0)
        {
            return 1.0e20f;
        }
        const float3 g = ToGrid01(pos01, info);
        int xi = floorToInt(g.x + 0.5f);
        int yi = floorToInt(g.y + 0.5f);
        int zi = floorToInt(g.z + 0.5f);
        xi = max(0, min(info.dim_.x - 1, xi));
        yi = max(0, min(info.dim_.y - 1, yi));
        zi = max(0, min(info.dim_.z - 1, zi));
        return info.volumePtr_[Index3D(xi, yi, zi, info.dim_)];
    }

    FUNCTION_MODIFIER_INLINE float SampleSDFTrilinear01(const float3 pos01, const SDFVolumeInfo& info)
    {
        if (!info.volumePtr_ || info.dim_.x * info.dim_.y * info.dim_.z == 0)
        {
            return 1.0e20f;
        }
        const float3 g = ToGrid01(pos01, info);
        int x0 = floorToInt(g.x);
        int y0 = floorToInt(g.y);
        int z0 = floorToInt(g.z);
        const int x1 = max(0, min(x0 + 1, info.dim_.x - 1));
        const int y1 = max(0, min(y0 + 1, info.dim_.y - 1));
        const int z1 = max(0, min(z0 + 1, info.dim_.z - 1));

        x0 = max(0, min(info.dim_.x - 1, x0));
        y0 = max(0, min(info.dim_.y - 1, y0));
        z0 = max(0, min(info.dim_.z - 1, z0));

        const float fx = fminf(fmaxf(g.x - static_cast<float>(x0), 0.0f), 1.0f);
        const float fy = fminf(fmaxf(g.y - static_cast<float>(y0), 0.0f), 1.0f);
        const float fz = fminf(fmaxf(g.z - static_cast<float>(z0), 0.0f), 1.0f);

        const float c000 = info.volumePtr_[Index3D(x0, y0, z0, info.dim_)];
        const float c100 = info.volumePtr_[Index3D(x1, y0, z0, info.dim_)];
        const float c010 = info.volumePtr_[Index3D(x0, y1, z0, info.dim_)];
        const float c110 = info.volumePtr_[Index3D(x1, y1, z0, info.dim_)];
        const float c001 = info.volumePtr_[Index3D(x0, y0, z1, info.dim_)];
        const float c101 = info.volumePtr_[Index3D(x1, y0, z1, info.dim_)];
        const float c011 = info.volumePtr_[Index3D(x0, y1, z1, info.dim_)];
        const float c111 = info.volumePtr_[Index3D(x1, y1, z1, info.dim_)];

        const float c00 = c000 * (1.0f - fx) + c100 * fx;
        const float c10 = c010 * (1.0f - fx) + c110 * fx;
        const float c01 = c001 * (1.0f - fx) + c101 * fx;
        const float c11 = c011 * (1.0f - fx) + c111 * fx;

        const float c0 = c00 * (1.0f - fy) + c10 * fy;
        const float c1 = c01 * (1.0f - fy) + c11 * fy;

        return c0 * (1.0f - fz) + c1 * fz;
    }
    /*
    FUNCTION_MODIFIER_INLINE float WorldToUnitDistance(float dWorld, const SDFVolumeInfo& info)
    {
        // Using isotropic scale based on bbox length min.
        return dWorld / info.unitScale_;
    }
    */
    FUNCTION_MODIFIER_INLINE float MinVoxelSize01(const SDFVolumeInfo& info)
    {
        const float vx = (info.dim_.x > 1) ? 1.0f / static_cast<float>(info.dim_.x - 1) : 1.0f;
        const float vy = (info.dim_.y > 1) ? 1.0f / static_cast<float>(info.dim_.y - 1) : 1.0f;
        const float vz = (info.dim_.z > 1) ? 1.0f / static_cast<float>(info.dim_.z - 1) : 1.0f;
        return fminf(vx, fminf(vy, vz));
    }
    
    FUNCTION_MODIFIER_INLINE float SampleSDFAdaptive01(
        float3 pos01,
        const SDFVolumeInfo& info)
    {
        //return SampleSDFTrilinear01(pos01, info);
        const float distanceNearest = SampleSDFNearest01(pos01, info);
        const float d = (fabsf(distanceNearest) > info.voxelSizeMax_) ? distanceNearest : SampleSDFTrilinear01(pos01, info);
        //return WorldToUnitDistance(d, info);
        return d;
    }

    struct SDFVolumeFunctor
    {
        SDFVolumeInfo info_ = {};
        // Construct with volume pointer and info known on host.
        FUNCTION_MODIFIER_INLINE SDFVolumeFunctor(const SDFVolumeInfo& info) : info_(info)
        {
        }
        // sdfInfo not using
        // Query in normalized 0..1^3 space.
        FUNCTION_MODIFIER_INLINE float operator()(const float3 pos01, [[maybe_unused]] const SDFInfo& sdfInfo) const
        {
            if (!info_.volumePtr_)
            {
                return 1.0e20f;
            }
            return SampleSDFAdaptive01(pos01 + 0.5f, info_);
        }
    };
}