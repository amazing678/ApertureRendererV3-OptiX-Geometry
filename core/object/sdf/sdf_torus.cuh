// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "sdf_base.cuh"

namespace sdf
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE AdditionalObjectInfo CreateTorus(const float scale = 1.0f, const float R = 0.375f, const float r = 0.125f)
    {
        return {ESDFType::SDF_TORUS, SDFInfo{float4{R * scale, r * scale, 0.0f, 0.0f}}};
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float SdfTorus(float3 u, float2 tNorm)
    {
        const float2 q = float2{ length(float2{ u.x, u.z }) - tNorm.x, u.y };
        return length(q) - tNorm.y;
    }
    struct SDFTorusFunctor
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE float operator()(const float3 position01, const SDFInfo& sdfInfo) const
        {
            return SdfTorus(position01, {sdfInfo.sdfInfo0_.x, sdfInfo.sdfInfo0_.y});
        }
    };
}
