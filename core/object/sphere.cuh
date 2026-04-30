// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "object.cuh"
#include <cuda_runtime.h>

namespace object
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float IntersectSphereBase(const float3 rayOrigin, const float3 rayDirection, const float radius, bool& bInside)
    {
        bInside = false;
        const float a = 1; // dot(rayDirection, rayDirection); // 
        const float b = dot(rayOrigin, rayDirection);
        const float c = dot(rayOrigin, rayOrigin) - radius * radius;

        const float disc = b * b - a * c;
        if (disc < 0.0f)
        {
            return inf();
        }

        const float s = sqrtf(disc);
        float t = -b - s;
        if (t <= 0.0f)
        {
            t = -b + s; // inner
            bInside = true;
        }
        if (t <= 0.0f)
        {
            return inf();
        }
        return t;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectSphereObject(const float3 rayOrigin, const float3 rayDirection, const SceneObject& object)
    {
        IntersectionContext ctx{};
        const float3 localOrigin = object.worldToObject_ * (rayOrigin - object.center_);
        const float3 localDirection = normalize(object.worldToObject_ * rayDirection);

        const float r = max3(object.extent_);

        bool bInside = false;
        const float tEnter = IntersectSphereBase(localOrigin, localDirection, r, bInside);

        if (isinf(tEnter))
        {
            return ctx;
        }
        const float tHit = tEnter;

        const float3 localPosition = localOrigin + localDirection * tHit;
        const float3 localNormal = normalize(localPosition) * (bInside ? -1.0f : 1.0f);

        const float3 worldPosition = object.objectToWorld_ * localPosition + object.center_;
        const float3 worldNormal = normalize(object.objectToWorld_ * localNormal);

        ctx.bHit_ = true;
        ctx.distance_ = tHit;
        ctx.hitLocalPosition_ = localPosition;
        ctx.hitLocalNormal_ = localNormal;
        ctx.hitPosition_ = worldPosition;
        ctx.hitNormal_ = worldNormal;
    
        ctx.gbuffer_ = object.material_.texture_.baseTexture_;
        ctx.gbuffer_.normal_ = worldNormal;
        return ctx;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE bool OverlapSphereBase(const float3 rayOrigin, const float radius)
    {
        return dot(rayOrigin, rayOrigin) < radius * radius;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapSphereObject(const float3 rayOrigin, const SceneObject& object)
    {
        OverlapContext ctx{};
        const float3 localRayOrigin = object.worldToObject_ * (rayOrigin - object.center_);
        const float r = max3(object.extent_);
        ctx.bOverlap_ = OverlapSphereBase(localRayOrigin, r);
        ctx.overlapPosition_ = rayOrigin;
        return ctx;
    }
}
