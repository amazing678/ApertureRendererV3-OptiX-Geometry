// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "object/AABB.cuh"
#include <cuda_runtime.h>

#include "sdf_volume.cuh"

namespace sdf
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 ToUnitSpace(const float3 pLocal, const float3 extent)
    {
        const float ex = fmaxf(extent.x, 1e-8f);
        const float ey = fmaxf(extent.y, 1e-8f);
        const float ez = fmaxf(extent.z, 1e-8f);
        // [-extent, +extent] -> [-0.5, +0.5]
        return make_float3( pLocal.x / (2.0f * ex),
                            pLocal.y / (2.0f * ey),
                            pLocal.z / (2.0f * ez) );
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float LocalScaleMin(const float3 extent)
    {
        return 2.0f * fminf(extent.x, fminf(extent.y, extent.z));
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 UnitNormalToLocal(const float3 n_unit, const float3 extent)
    {
        // n_local ~ S^{-T} n_unit, S=2*diag(extent)
        const float ex = fmaxf(extent.x, 1e-8f);
        const float ey = fmaxf(extent.y, 1e-8f);
        const float ez = fmaxf(extent.z, 1e-8f);
        float3 n = make_float3(n_unit.x / (2.0f * ex),
                               n_unit.y / (2.0f * ey),
                               n_unit.z / (2.0f * ez));
        return normalize(n);
    }

    template<typename SDFFunctor>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 sdfNormalUnit(const float3 position, const SDFInfo& sdfInfo, const float hUnit, const SDFFunctor& sdf)
    {
        const float3 ex = make_float3(hUnit, 0.0f, 0.0f);
        const float3 ey = make_float3(0.0f, hUnit, 0.0f);
        const float3 ez = make_float3(0.0f, 0.0f, hUnit);

        const float dx = sdf(position + ex, sdfInfo) - sdf(position - ex, sdfInfo);
        const float dy = sdf(position + ey, sdfInfo) - sdf(position - ey, sdfInfo);
        const float dz = sdf(position + ez, sdfInfo) - sdf(position - ez, sdfInfo);
        return normalize(make_float3(dx, dy, dz));
    }

    template<typename SDFFunctor, bool bNeedScale = false>
    [[nodiscard]] FUNCTION_MODIFIER bool RaymarchLocal(
        const int depth,
        const float3 localRayOrigin,
        const float3 localRayDirection,
        const float tEnter,
        const float tExit,
        const float3 extent,
        const SDFInfo& sdfInfo,
        float* outTHit,
        float3* outLocalPos,
        const SDFFunctor& sdf)
    {
        const float currentEPS = SDF_EPS(depth);
        const float minStep = currentEPS * 0.5f;
        const bool bStartInside = sdf(ToUnitSpace(localRayOrigin, extent), sdfInfo) <= 0.0f;
        float t = fmaxf(0.0f, tEnter);
        float tLast = t;
        const float sMin = LocalScaleMin(extent);
        [[maybe_unused]] const float epsUnit = currentEPS / fmaxf(sMin, 1e-8f);
        SDF_CUDA_UNROLL
        for (int i = 0; i < SDF_STEPS; ++i)
        {
            tLast = t;
            const float3 pL = localRayOrigin + localRayDirection * t;
            const float3 u = ToUnitSpace(pL, extent);
            const float dUnit = sdf(u, sdfInfo);
            const float dLocal = dUnit * sMin;
            const float realDistance = fabsf(dLocal);
            // started from outside -> must cross into inner
            // started from inside -> must cross out to outer 
            if (bStartInside ? (dLocal > 0 && dLocal < currentEPS) : (dLocal < 0 && -dLocal < currentEPS))
            {
                if (outTHit)
                {
                    *outTHit = tLast;
                }
                if (outLocalPos)
                {
                    *outLocalPos = pL;
                }
                return true;
            }
            if constexpr (bNeedScale)
            {
                const float advance = fmaxf(realDistance * SDF_VOLUME_STEP_SCALE, minStep);
                t += advance;
            }
            else
            {
                const float advance = fmaxf(realDistance, minStep);
                t += advance;
            }
            if (t > tExit)
            {
                break;
            }
        }
        return false;
    }

    template<typename SDFFunctor, bool bNeedScale = false>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectSDFObject(const int depth, const float3 rayOrigin, const float3 rayDirection, const SceneObject& object, const SDFFunctor& sdf)
    {
        const SDFInfo& sdfInfo = object.additionalObjectInfo_.sdfInfo_;
        IntersectionContext ctx{};

        const float3 localRayOrigin = object.worldToObject_ * (rayOrigin - object.center_);
        const float3 localRayDirection = normalize(object.worldToObject_ * rayDirection);

        const float3 boundMin = -object.extent_;
        const float3 boundMax = object.extent_;
        const float2 tr = object::IntersectAABBBase(localRayOrigin, localRayDirection, boundMin, boundMax);
        const float  tEnter = tr.x, tExit = tr.y;

        if (tEnter > tExit)
        {
            return ctx;
        }
        if (tExit  <= 0.0f)
        {
            return ctx;
        }

        float tHit = 0.0f;
        float3 localPos = make_float3(0.0f, 0.0f, 0.0f);
        const bool hit = RaymarchLocal<SDFFunctor, bNeedScale>(depth, localRayOrigin, localRayDirection, tEnter, tExit, object.extent_, sdfInfo, &tHit, &localPos, sdf);
        if (!hit)
        {
            return ctx;
        }
        const float sMin = LocalScaleMin(object.extent_);
        const float hUnit = SDF_EPS(depth) / fmaxf(sMin, 1e-8f);
        const float3 uHit = ToUnitSpace(localPos, object.extent_);
        const float3 nUnit = sdfNormalUnit(uHit, sdfInfo, hUnit, sdf);
        const float3 nLocal = UnitNormalToLocal(nUnit, object.extent_);
        const float3 worldPos = object.objectToWorld_ * localPos + object.center_;
        const float3 worldNormal = normalize(object.objectToWorld_ * nLocal);

        ctx.bHit_ = true;
        ctx.distance_ = tHit;
        ctx.hitLocalPosition_ = localPos;
        ctx.hitLocalNormal_ = nLocal;
        ctx.hitPosition_ = worldPos;
        ctx.hitNormal_ = worldNormal;
        ctx.gbuffer_ = object.material_.texture_.baseTexture_;
        ctx.gbuffer_.normal_ = worldNormal;
        ctx.objectIndex_ = object.objectIndex_;
        ctx.lightIndex_ = object.lightIndex_;
        return ctx;
    }

    template<typename SDFFunctor>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapSDFObject(const float3 rayOrigin, const SceneObject& object, const SDFFunctor& sdf)
    {
        const SDFInfo& sdfInfo = object.additionalObjectInfo_.sdfInfo_;
        OverlapContext ctx{};
        
        const float3 pLocal = object.worldToObject_ * (rayOrigin - object.center_);
        const float3 u = ToUnitSpace(pLocal, object.extent_);
        ctx.bOverlap_ = (sdf(u, sdfInfo) <= 0.0f);
        ctx.overlapPosition_= rayOrigin;
        ctx.objectIndex_ = object.objectIndex_;
        return ctx;
    }

    struct SDFTorusFunctor;
    struct SDFDiamondFunctor;

    extern template float3 sdfNormalUnit<SDFTorusFunctor>(float3, const SDFInfo&, float, const SDFTorusFunctor&);
    extern template bool RaymarchLocal<SDFTorusFunctor>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFTorusFunctor&);
    extern template IntersectionContext IntersectSDFObject<SDFTorusFunctor>(const int, float3, float3, const SceneObject&, const SDFTorusFunctor&);
    extern template OverlapContext OverlapSDFObject<SDFTorusFunctor>(float3, const SceneObject&, const SDFTorusFunctor&);

    extern template float3 sdfNormalUnit<SDFDiamondFunctor>(float3, const SDFInfo&, float, const SDFDiamondFunctor&);
    extern template bool RaymarchLocal<SDFDiamondFunctor>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFDiamondFunctor&);
    extern template IntersectionContext IntersectSDFObject<SDFDiamondFunctor>(const int, float3, float3, const SceneObject&, const SDFDiamondFunctor&);
    extern template OverlapContext OverlapSDFObject<SDFDiamondFunctor>(float3, const SceneObject&, const SDFDiamondFunctor&);

    extern template float3 sdfNormalUnit<SDFVolumeFunctor>(float3, const SDFInfo&, float, const SDFVolumeFunctor&);
    extern template bool RaymarchLocal<SDFVolumeFunctor, true>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFVolumeFunctor&);
    extern template IntersectionContext IntersectSDFObject<SDFVolumeFunctor, true>(const int, float3, float3, const SceneObject&, const SDFVolumeFunctor&);
    extern template OverlapContext OverlapSDFObject<SDFVolumeFunctor>(float3, const SceneObject&, const SDFVolumeFunctor&);
}
