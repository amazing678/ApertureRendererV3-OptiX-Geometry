#pragma once
#include <cuda_runtime.h>

#include "vector.cuh"

namespace spectrum
{
    static constexpr float wavelengthF = 486.1327f;
    static constexpr float wavelengthD = 587.5618f;
    static constexpr float wavelengthC = 656.2725f;
    static constexpr float invF = 1.0f / wavelengthF;
    static constexpr float invC = 1.0f / wavelengthC;
    static constexpr float invD = 1.0f / wavelengthD;

    // nd: IOR, Vd: abbe number
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float IORLinearKFromAbbe(const float nd, const float Vd)
    {
        // Guard against invalid inputs
        if (!(Vd > 0.0f))
        {
            return 0.0f;
        }
        return (nd - 1.0f) / (Vd * (invF - invC));
    }
    
    // nd: IOR, Vd: abbe number
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float IORFromAbbeNM(const float lambda_nm, const float nd, const float Vd)
    {
        if (!(lambda_nm > 0.0f)) return nd; // simple guard
        const float K = IORLinearKFromAbbe(nd, Vd);
        const float invL = 1.0f / lambda_nm;
        return nd + K * (invL - invD);
    }

    // nd: IOR, Vd: abbe number
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 IORFromAbbeNM(const float3 lambda_nm, const float nd, const float Vd)
    {
        const float K = IORLinearKFromAbbe(nd, Vd);
        const float3 invL = make_f3(1.0f / lambda_nm.x, 1.0f / lambda_nm.y, 1.0f / lambda_nm.z);
        return make_f3(
            nd + K * (invL.x - invD),
            nd + K * (invL.y - invD),
            nd + K * (invL.z - invD)
        );
    }
    static constexpr float deltaU = (invF - invC);
    [[nodiscard]] FUNCTION_MODIFIER_INLINE
    float SigmaAFromTiltNM(float sigmaA, float lambda_nm, float sigmaTilt, float tiltGain = 1.0f)
    {
        const float u  = 1.0f / fmaxf(lambda_nm, 1.0e-6f);
        const float du = (u - invD) / deltaU;
        const float sA = sigmaA * (1.0f + tiltGain * sigmaTilt * du);
        return fmaxf(0.0f, sA);
    }
}
