// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "render/object.cuh"

namespace object
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float2 IntersectAABBBase(const float3 rayOrigin, const float3 rayDirection, const float3 boundMin, const float3 boundMax)
    {
        //float3 div = { 1.0f / rayDirection.x, 1.0f / rayDirection.y, 1.0f / rayDirection.z };
        float3 div = invSafe(rayDirection);

        float3 t1 = (boundMin - rayOrigin) * div;
        float3 t2 = (boundMax - rayOrigin) * div;

        float3 tMin2 = min(t1, t2);
        float3 tMax2 = max(t1, t2);

        //const float tMin = fmaxf(fmaxf(tMin2.x, tMin2.y), fmaxf(tMin2.z, 0.0f));
        //const float tMax = fminf(fminf(tMax2.x, tMax2.y), fminf(tMax2.z, inf()));
        const float tMin = fmaxf(fmaxf(tMin2.x, tMin2.y), tMin2.z);
        const float tMax = fminf(fminf(tMax2.x, tMax2.y), tMax2.z);

        return { tMin ,tMax };
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectAABBObject(const float3 rayOrigin, const float3 rayDirection, const SceneObject& object)
    {
        IntersectionContext ctx{};
    
        const float3 localRayOrigin = object.worldToObject_ * (rayOrigin - object.center_);
        const float3 localRayDirection = normalize(object.worldToObject_ * rayDirection);

        const float3 boundMin = -object.extent_;
        const float3 boundMax =  object.extent_;

        const float2 tRange = IntersectAABBBase(localRayOrigin, localRayDirection, boundMin, boundMax);
        const float tEnter = tRange.x;
        const float tExit  = tRange.y;

        if (tEnter > tExit)
        {
            return ctx;
        }
        if (tExit <= 0.0f)
        {
            return ctx;
        }
        const bool  bInside = (tEnter < 0.0f);
        const float tHit = bInside ? tExit : tEnter;
    
        const float3 localPosition = localRayOrigin + localRayDirection * tHit;
        const float3 worldPosition = object.objectToWorld_ * localPosition + object.center_;

        const float rx = fabsf(localPosition.x) / object.extent_.x;
        const float ry = fabsf(localPosition.y) / object.extent_.y;
        const float rz = fabsf(localPosition.z) / object.extent_.z;

        float3 localNormal;
        if (rx >= ry && rx >= rz)
        {
            localNormal = float3{(localPosition.x > 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f};
        }
        else if (ry >= rz)
        {
            localNormal = float3{0.0f, (localPosition.y > 0.0f) ? 1.0f : -1.0f, 0.0f};
        }
        else
        {
            localNormal = float3{0.0f, 0.0f, (localPosition.z > 0.0f) ? 1.0f : -1.0f};
        }
        if (bInside)
        {
            localNormal = -localNormal;
        }
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

    [[nodiscard]] FUNCTION_MODIFIER_INLINE bool OverlapAABBBase(const float3 rayOrigin, const float3 boundMin, const float3 boundMax)
    {
        return boundMin.x < rayOrigin.x && boundMin.y < rayOrigin.y && boundMin.z < rayOrigin.z
        && boundMax.x > rayOrigin.x && boundMax.y > rayOrigin.y && boundMax.z > rayOrigin.z;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapAABBObject(const float3 rayOrigin, const SceneObject& object)
    {
        OverlapContext ctx{};
        const float3 localRayOrigin = object.worldToObject_ * (rayOrigin - object.center_);
        const float3 boundMin = -object.extent_;
        const float3 boundMax =  object.extent_;
        ctx.bOverlap_ = OverlapAABBBase(localRayOrigin, boundMin, boundMax);
        ctx.overlapPosition_ = rayOrigin;
        return ctx;
    }
}
