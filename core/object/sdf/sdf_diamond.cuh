// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "sdf_base.cuh"

namespace sdf
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE AdditionalObjectInfo CreateDiamond(const float scale = 1.0f, const float d = 0.94f, const float bias = 0.5f, const float tableZ = 0.3f)
    {
        return {ESDFType::SDF_UNDEFINED, SDFInfo{float4{d, bias, tableZ, scale * 0.5f}}};
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float PlaneSDF(const float3 p, const float3 n, const float d)
    {
        const float len = fmaxf(sqrtf(n.x * n.x + n.y * n.y + n.z * n.z), 1e-8f);
        return (p.x * n.x + p.y * n.y + p.z * n.z - d) / len;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE float SdfDiamondUnit(const float3 u, const float4 sdfInfo)
    {
        const float d = (sdfInfo.x != 0.0f) ? sdfInfo.x : 0.94f;
        const float bias = (sdfInfo.y != 0.0f) ? sdfInfo.y : 0.5f;
        const float tableZ = (sdfInfo.z != 0.0f) ? sdfInfo.z : 0.3f;
        const float scale = (sdfInfo.w != 0.0f) ? sdfInfo.w : 1.0f;
        const float scaleInv = 1.0f / scale;

        const float3 p = make_float3(u.x * scaleInv, u.y * scaleInv, u.z * scaleInv);

        constexpr float af2 = 4.0f / CUDART_PI_F;
        const float s = atan2f(p.y, p.x);
        const float sf = floorf(s * af2 + bias) / af2;
        const float sf2 = floorf(s * af2) / af2;

        const float csf = cosf(sf);
        const float ssf = sinf(sf);
        const float cs = cosf(s);
        const float ss = sinf(s);
        const float csf1 = cosf(sf + 0.21f);
        const float ssf1 = sinf(sf + 0.21f);
        const float csf2 = cosf(sf - 0.21f);
        const float ssf2 = sinf(sf - 0.21f);
        const float csf8 = cosf(sf2 + 0.393f);
        const float ssf8 = sinf(sf2 + 0.393f);
        float dMax = -1e20f;
        // Crown, bezel facets
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf, ssf, 1.444f), d));
        // Pavillon, pavillon facets
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf, ssf, -1.072f), d));
        // Table
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(0.0f, 0.0f, 1.0f), tableZ));
        // Cutlet
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(0.0f, 0.0f, -1.0f), 0.865f));
        // Girdle
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(cs, ss, 0.0f), 0.911f));
        // Pavillon, lower-girdle facets
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf1, ssf1, -1.02f), 0.9193f));
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf2, ssf2, -1.02f), 0.9193f));
        // Crown, upper-girdle facets
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf2, ssf2, 1.03f), 0.912f));
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf1, ssf1, 1.03f), 0.912f));
        // Crown, star facets
        dMax = fmaxf(dMax, PlaneSDF(p, make_float3(csf8, ssf8, 2.21f), 1.131f));
        return dMax * scale;
    }

    struct SDFDiamondFunctor
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE float operator()(const float3 position01, const SDFInfo& sdfInfo) const
        {
            return SdfDiamondUnit(position01, sdfInfo.sdfInfo0_);
        }
    };
}
