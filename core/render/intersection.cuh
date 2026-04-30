// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "object.cuh"
#include "object/AABB.cuh"
#include "object/Sphere.cuh"
#include "object/sdf/sdf_master.cuh"
#include <cuda_runtime.h>

#include "object/sdf/sdf_torus.cuh"

namespace intersection
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectTriangleObject(const float3 rayOrigin, const float3 rayDirection, const SceneObject& object)
    {
        IntersectionContext result = {};

        //如果物体不可见，直接返回未命中 (Miss)
        if (!object.bVisible_) return result;

        // 1. 获取三角形局部顶点，并转换到世界空间
        const Vertex& v0 = object.additionalObjectInfo_.triangleInfo_.v0_;
        const Vertex& v1 = object.additionalObjectInfo_.triangleInfo_.v1_;
        const Vertex& v2 = object.additionalObjectInfo_.triangleInfo_.v2_;

        const float3 p0 = object.objectToWorld_ * v0.position_ + object.center_;
        const float3 p1 = object.objectToWorld_ * v1.position_ + object.center_;
        const float3 p2 = object.objectToWorld_ * v2.position_ + object.center_;

        // 2. Möller-Trumbore 算法核心
        const float3 e1 = p1 - p0;
        const float3 e2 = p2 - p0;
        const float3 h = cross(rayDirection, e2);
        const float a = dot(e1, h);

        // 如果 a 接近 0，说明光线与三角形平行 (Miss)
        if (a > -1e-6f && a < 1e-6f) return result;

        const float f = 1.0f / a;
        const float3 s = rayOrigin - p0;

        // 3. 计算重心坐标 U
        const float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) return result;

        const float3 q = cross(s, e1);

        // 4. 计算重心坐标 V
        const float v = f * dot(rayDirection, q);
        if (v < 0.0f || u + v > 1.0f) return result;

        // 5. 计算射线距离 T
        const float t = f * dot(e2, q);

        // 确保交点在光线前方 (1e-5f 防止表面自相交引发的阴影粉刺)
        if (t > 1e-5f)
        {
            result.bHit_ = true;
            result.distance_ = t;
            result.hitPosition_ = rayOrigin + rayDirection * t;

            result.gbuffer_.albedoTex_ = object.material_.texture_.baseTexture_.albedoTex_;

            // 6.核心：通过重心坐标 (u, v, w) 进行平滑插值
            const float w = 1.0f - u - v;

            // 插值 UV 坐标
            result.uv_ = make_float2(
                v0.uv_.x * w + v1.uv_.x * u + v2.uv_.x * v,
                v0.uv_.y * w + v1.uv_.y * u + v2.uv_.y * v
            );

            // 插值法线 (先转换到世界空间，再归一化)
            const float3 n0 = object.objectToWorld_ * v0.normal_;
            const float3 n1 = object.objectToWorld_ * v1.normal_;
            const float3 n2 = object.objectToWorld_ * v2.normal_;
            float3 interpolatedNormal = normalize(n0 * w + n1 * u + n2 * v);

            // 7.核心：内外判断与法线翻转 (完美支持玻璃/水体材质)
            // 如果光线方向和法线方向相同（点乘 > 0），说明光线正从物体内部穿出
            result.isInside_ = dot(rayDirection, interpolatedNormal) > 0.0f;
            result.hitNormal_ = result.isInside_ ? -interpolatedNormal : interpolatedNormal;

            return result;
        }

        return result;
    }


    // factory
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectBase(const int depth, float3 rayOrigin, float3 rayDirection, const SceneObject& object)
    {
        if(object.type_ == EObjectType::OBJ_CUBE)
        {
            return object::IntersectAABBObject(rayOrigin, rayDirection, object);
        }
        else if(object.type_ == EObjectType::OBJ_SPHERE)
        {
            return object::IntersectSphereObject(rayOrigin, rayDirection, object);
        }
        else if(object.type_ == EObjectType::OBJ_SDF)
        {
            return sdf::IntersectSDFObject(depth, rayOrigin, rayDirection, object);
        }
        //三角形部分
        else if (object.type_ == EObjectType::OBJ_TRIANGLE)
        {
            return IntersectTriangleObject(rayOrigin, rayDirection, object);
        }
        return {};
    }

    // overlap factory
    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapBase(float3 rayOrigin, const SceneObject& object)
    {
        if(object.type_ == EObjectType::OBJ_CUBE)
        {
            return object::OverlapAABBObject(rayOrigin, object);
        }
        else if(object.type_ == EObjectType::OBJ_SPHERE)
        {
            return object::OverlapSphereObject(rayOrigin, object);
        }
        else if(object.type_ == EObjectType::OBJ_SDF)
        {
            return sdf::OverlapSDFObject(rayOrigin, object);
        }
        //三角形无线薄时返回空
        else if (object.type_ == EObjectType::OBJ_TRIANGLE)
        {
            return {};
        }
        return {};
    }

    // intersector
    // [min, max)
    struct SceneObjectsIntersector
    {
        const SceneObject* sceneObjects_; // alias
        int sceneCount_ = -1;
    
        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(
            const int depth,
            const int objectIndex,
            const float3 rayOrigin, const float3 rayDirection,
            const float tMin, const float tMax,
            IntersectionContext* __restrict__ outIntersectionContext = nullptr  // need full intersection info, fill all info here
        ) const
        {
            const SceneObject* __restrict__ restrictSceneObjects = sceneObjects_;
            const SceneObject& currentObject = restrictSceneObjects[objectIndex];
            const IntersectionContext intersectionContext = IntersectBase(depth, rayOrigin, rayDirection, currentObject);
            if (!intersectionContext.bHit_)
            {
                return false;
            }
            if (intersectionContext.distance_ < tMin || intersectionContext.distance_ >= tMax)
            {
                return false;
            }
            if (outIntersectionContext)
            {
                *outIntersectionContext = intersectionContext;
                outIntersectionContext->objectIndex_ = objectIndex;
                outIntersectionContext->lightIndex_ = currentObject.lightIndex_;
            }
            return true;
        }
    };

    // master
    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext Intersect(
        const int depth,
        const float3 rayOrigin,
        const float3 rayDirection,
        const Intersector& intersector,
        const int soloObjectIndex,
        float tMin,
        float tMax = inf())
    {
        if (soloObjectIndex >= 0)
        {
            IntersectionContext result{};
            if (intersector(depth, soloObjectIndex, rayOrigin, rayDirection, tMin, tMax, &result))
            {
                return result;
            }
            return result; // miss
        }
        IntersectionContext result{};
        result.distance_ = tMax;
        for (int i = 0; i < intersector.sceneCount_; ++i)
        {
            IntersectionContext hitResult{};
            if (intersector(depth, i, rayOrigin, rayDirection, tMin, result.distance_, &hitResult))
            {
                result = hitResult;
            }
        }
        return result;
    }

    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext AnyHitSegment(const int depth, const float3 sourcePoint, const float3 targetPoint, const Intersector& intersector)
    {
        float3 direction = targetPoint - sourcePoint;
        const float distance = length(direction);
        if (distance <= 0.0f)
        {
            return {};
        }
        direction = direction / distance;
        const float eps = INTERSECT_EPS(depth) * fmaxf(1.0f, fmaxf(length(sourcePoint), length(targetPoint)));
        const float tMin = eps;
        const float tMax = fmaxf(0.0f, distance - eps);
        if (tMax <= 0.0f)
        {
            return {};
        }
        for (int i = 0; i < intersector.sceneCount_; ++i)
        {
            IntersectionContext tmp{};
            if (intersector(depth, i, sourcePoint, direction, tMin, tMax, &tmp))
            {
                return tmp;
            }
        }
        return {};
    }

    template <bool bAnyHit, typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER IntersectionContext TraverseBVH(
        const int depth,
        const float3 rayOrigin,
        const float3 rayDirection,
        const Intersector& intersector,
        const BVHNode* __restrict__ nodes,
        const int nodeCount,
        const int* __restrict__ objectIndices,
        float tMin,
        float tMax = inf()
        )
    {
        IntersectionContext result = {};
        if (nodeCount == 0)
        {
            return result;
        }
        result.distance_ = tMax;

        constexpr int STACK_CAP = 128;
        int stack[STACK_CAP];
        int sp = 0;
        auto PUSH = [&](int n) 
        {
            if (sp < STACK_CAP)
            {
                stack[sp++] = n;
            }
#ifdef DEBUG
            else assert(false && "BVH traversal stack overflow");
#endif
        };
        PUSH(0);
        const float3 invDirection = invSafe(rayDirection);
        while (sp)
        {
            const int currentNodeIndex = stack[--sp];
            const BVHNode& currentNode = nodes[currentNodeIndex];
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
            const float4 prefetchMinLeftRight = currentNode.boundMinLeftRight_;
            const float4 prefetchMaxFirstCount = currentNode.boundMaxFirstCount_;
            const float3 currentNodeMin = xyz(prefetchMinLeftRight);
            const float3 currentNodeMax = xyz(prefetchMaxFirstCount);
            uint16_t currentNodeLeft;
            uint16_t currentNodeRight;
            uint16_t currentNodeFirst;
            uint16_t currentNodeCount;
            Unpack16x2(floatAsUint(prefetchMinLeftRight.w), currentNodeLeft, currentNodeRight);
            Unpack16x2(floatAsUint(prefetchMaxFirstCount.w), currentNodeFirst, currentNodeCount);
#else
            const float3 currentNodeMin = currentNode.boundMin_;
            const float3 currentNodeMax = currentNode.boundMax_;
            const int currentNodeLeft = currentNode.left_;
            const int currentNodeRight = currentNode.right_;
            const int currentNodeFirst = currentNode.first_;
            const int currentNodeCount = currentNode.count_;
#endif
            if (!FastAABBHit(rayOrigin, invDirection, currentNodeMin, currentNodeMax, tMin, result.distance_))
            {
                continue;
            }
            if (currentNodeCount > 0)
            {
                for (int i = 0; i < currentNodeCount; ++i)
                {
                    const int objectIndex = objectIndices[currentNodeFirst + i];
                    if (intersector(depth, objectIndex, rayOrigin, rayDirection, tMin, result.distance_, &result))
                    {
                        if constexpr (bAnyHit)
                        {
                            return result;
                        }
                    }
                }
            }
            else
            {
                const BVHNode& left = nodes[currentNodeLeft];
                const BVHNode& right = nodes[currentNodeRight];
#ifdef BVH_USE_SORT_STACK
                float tL = 0.0f;
                float tR = 0.0f;
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
                const bool hitL = FastAABBHitWithEnter(rayOrigin, invDirection, xyz(left.boundMinLeftRight_),  xyz(left.boundMaxFirstCount_),  tMin, result.distance_, tL);
                const bool hitR = FastAABBHitWithEnter(rayOrigin, invDirection, xyz(right.boundMinLeftRight_), xyz(right.boundMaxFirstCount_), tMin, result.distance_, tR);
#else
                const bool hitL = FastAABBHitWithEnter(rayOrigin, invDirection, left.boundMin_,  left.boundMax_,  tMin, result.distance_, tL);
                const bool hitR = FastAABBHitWithEnter(rayOrigin, invDirection, right.boundMin_, right.boundMax_, tMin, result.distance_, tR);
#endif
                if (hitL && hitR)
                {
                    if (tL > tR)
                    {
                        PUSH(currentNodeLeft);
                        PUSH(currentNodeRight);
                    }
                    else
                    {
                        PUSH(currentNodeRight);
                        PUSH(currentNodeLeft);
                    }
                }
                else if (hitL)
                {
                    PUSH(currentNodeLeft);
                }
                else if (hitR)
                {
                    PUSH(currentNodeRight);
                }
#else
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
                const float3 leftCenter = AABBCenter(xyz(left.boundMinLeftRight_), xyz(left.boundMaxFirstCount_));
                const float3 rightCenter = AABBCenter(xyz(right.boundMinLeftRight_), xyz(right.boundMaxFirstCount_));
#else
                const float3 leftCenter = AABBCenter(left.boundMin_, left.boundMax_);
                const float3 rightCenter = AABBCenter(right.boundMin_, right.boundMax_);
#endif
                const float distanceLeft = dot(leftCenter  - rayOrigin,  rayDirection);
                const float distanceRight = dot(rightCenter - rayOrigin,  rayDirection);
                if (distanceLeft > distanceRight)
                {
                    PUSH(currentNodeLeft);
                    PUSH(currentNodeRight);
                }
                else
                {
                    PUSH(currentNodeRight);
                    PUSH(currentNodeLeft);
                }
#endif
            }
        }
        return result;
    }

    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext AnyHitSegmentBVH(
        const int depth,
        const float3 sourcePoint, const float3 targetPoint, const Intersector& intersector,
        const BVHNode* __restrict__ nodes,
        const int nodeCount,
        const int* __restrict__ objectIndices)
    {
        float3 direction = targetPoint - sourcePoint;
        const float distance = length(direction);
        if (distance <= 0.0f)
        {
            return {};
        }
        direction = direction / distance;
        const float eps = INTERSECT_EPS(depth) * fmaxf(1.0f, fmaxf(length(sourcePoint), length(targetPoint)));
        const float tMin = 0.0f;
        const float tMax = fmaxf(0.0f, distance - eps);
        if (tMax <= 0.0f)
        {
            return {};
        }
        const IntersectionContext intersectionContext = TraverseBVH<true>( // AnyHit = true
            depth, sourcePoint, direction, intersector, nodes, nodeCount, objectIndices, tMin, tMax);
        return intersectionContext;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext Overlap(float3 rayOrigin, const SceneObject* __restrict__ sceneObjects, const int sceneObjectCounts)
    {
        OverlapContext result = {};
        for(int i = 0; i < sceneObjectCounts; i++)
        {
            const SceneObject& currentObject = sceneObjects[i];
            const OverlapContext overlapResult = OverlapBase(rayOrigin, currentObject);
            if (overlapResult.bOverlap_)
            {
                result = overlapResult;
                result.objectIndex_ = i;
                return result;
            }
        }
        return result;
    }
}
