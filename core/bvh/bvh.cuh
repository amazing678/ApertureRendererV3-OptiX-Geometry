// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <functional>

#include "vector.cuh"
#include "config.cuh"

struct PrimitiveProxy
{
    float3 boundMin_;
    float3 boundMax_;
    float3 center_;
};

struct alignas(16) BVHNode
{
#ifdef BVH_USE_16_BITS_NODE
#ifdef BVH_PACK_NODE
    float4 boundMinLeftRight_; // w pack left_, right_
    float4 boundMaxFirstCount_; // w pack first_, count_
#else
    float3 boundMin_;
    float3 boundMax_;
    uint16_t left_; // int16 is faster
    uint16_t right_;
    uint16_t first_;
    uint16_t count_;
#endif
#else
    float3 boundMin_;
    float3 boundMax_;
    int left_;
    int right_;
    int first_;
    int count_;
#endif
};

struct BVH 
{
    BVHNode* nodes_ = nullptr;
    int* objectIndices_ = nullptr;
    int nodeCount_ = 0;
    int objectCount_ = 0;
};

struct BVHHost
{
    std::vector<BVHNode> nodes_ = {};
    std::vector<int> objectIndices_ = {};
};

#if defined(BVH_USE_16_BITS_NODE)
#define BVH_NODE_16 1
#else
#define BVH_NODE_16 0
#endif

#if defined(BVH_PACK_NODE)
#define BVH_NODE_PACK 1
#else
#define BVH_NODE_PACK 0
#endif

#if (BVH_NODE_16 && BVH_NODE_PACK)
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes in 16bit+packed layout");
#elif (BVH_NODE_16 && !BVH_NODE_PACK)
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes in 16bit layout");
#else
static_assert(sizeof(BVHNode) == 48 || sizeof(BVHNode) == 40, "Unexpected BVHNode size in 32bit layout");
#endif

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 AABBMin(const float3 a, const float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 AABBMax(const float3 a, const float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float AABBArea(const float3 boundMin, const float3 boundMax)
{
    const float3 d = make_float3(fmaxf(0.0f, boundMax.x - boundMin.x), fmaxf(0.0f, boundMax.y - boundMin.y), fmaxf(0.0f, boundMax.z - boundMin.z));
    return 2.0f * (d.x*d.y + d.y*d.z + d.z*d.x);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 AABBCenter(const float3 boundMin, const float3 boundMax)
{
    return {0.5f * (boundMin.x + boundMax.x), 0.5f * (boundMin.y + boundMax.y), 0.5f * (boundMin.z + boundMax.z)};
}

struct BVHBinData
{
    float3 boundMin_;
    float3 boundMax_;
    int count_;
    FUNCTION_MODIFIER_INLINE void reset()
    {
        boundMin_ = { FLT_MAX,  FLT_MAX,  FLT_MAX};
        boundMax_ = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        count_ = 0;
    }
    FUNCTION_MODIFIER_INLINE void add(const float3& mn, const float3& mx)
    {
        boundMin_ = AABBMin(boundMin_, mn);
        boundMax_ = AABBMax(boundMax_, mx);
        count_++;
    }
};

struct BoundsAccumulator
{
    float3 boundMin_;
    float3 boundMax_; // primitive bounds
    float3 centroidMin_;
    float3 centroidMax_; // centroid bounds
    FUNCTION_MODIFIER_INLINE BoundsAccumulator()
    {
        boundMin_ = { FLT_MAX,  FLT_MAX,  FLT_MAX};
        boundMax_ = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        centroidMin_ = { FLT_MAX,  FLT_MAX,  FLT_MAX};
        centroidMax_ = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    }
    FUNCTION_MODIFIER_INLINE void add(const PrimitiveProxy& p)
    {
        boundMin_ = AABBMin(boundMin_, p.boundMin_);
        boundMax_ = AABBMax(boundMax_, p.boundMax_);
        centroidMin_ = AABBMin(centroidMin_, p.center_);
        centroidMax_ = AABBMax(centroidMax_, p.center_);
    }
};

[[nodiscard]] FUNCTION_MODIFIER_HOST_INLINE BVHHost BuildBVH(const std::vector<PrimitiveProxy>& proxy, int leafSize = 8, int binCount = 12)
{
#ifdef BVH_USE_16_BITS_NODE
    assert(proxy.size() < 32768);
#endif
    BVHHost result;
    if (proxy.empty())
    {
        result.nodes_ = {};
        result.objectIndices_ = {};
        return result;
    }
    binCount = std::max(4, std::min(64, binCount));
    leafSize = std::max(1, std::min(16, leafSize));

    const int primitiveCount = static_cast<int>(proxy.size());
    result.objectIndices_.resize(primitiveCount);
    for (int i = 0; i < proxy.size(); ++i)
    {
        result.objectIndices_[i] = i;
    }

    result.nodes_.reserve(2 * primitiveCount);

    auto computeBounds = [&proxy, &result](const int start, const int end)
    {
        BoundsAccumulator boundsAccumulator;
        for (int i = start; i < end; ++i)
        {
            boundsAccumulator.add(proxy[result.objectIndices_[i]]);
        }
        return boundsAccumulator;
    };

    std::function<int(int,int,int)> build = [&](const int start, const int end, const int depth) -> int
    {
        const int count = end - start;
        const int nodeIdx = static_cast<int>(result.nodes_.size());
        result.nodes_.emplace_back(); // placeholder
        BVHNode& node = result.nodes_.back();

        BoundsAccumulator boundsAccumulator = computeBounds(start, end);
        
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
        node.boundMinLeftRight_.x = boundsAccumulator.boundMin_.x;
        node.boundMinLeftRight_.y = boundsAccumulator.boundMin_.y;
        node.boundMinLeftRight_.z = boundsAccumulator.boundMin_.z;
        node.boundMaxFirstCount_.x = boundsAccumulator.boundMax_.x;
        node.boundMaxFirstCount_.y = boundsAccumulator.boundMax_.y;
        node.boundMaxFirstCount_.z = boundsAccumulator.boundMax_.z;
#else
        node.boundMin_ = boundsAccumulator.boundMin_;
        node.boundMax_ = boundsAccumulator.boundMax_;
#endif
        
        if (count <= leafSize)
        {
#ifdef BVH_USE_16_BITS_NODE
#ifdef BVH_PACK_NODE
            node.boundMinLeftRight_.w = uintAsFloat(Pack16x2(0, 0));
            node.boundMaxFirstCount_.w = uintAsFloat(Pack16x2(static_cast<uint16_t>(start), static_cast<uint16_t>(count)));
#else
            node.left_  = 0;
            node.right_ = 0;
            node.first_ = static_cast<uint16_t>(start);
            node.count_ = static_cast<uint16_t>(count);
#endif
#else
            node.left_  = -1;
            node.right_ = -1;
            node.first_ = start;
            node.count_ = count;
#endif
            return nodeIdx;
        }

        // Longest centroid axis
        const float3 centroidExtent = {boundsAccumulator.centroidMax_.x - boundsAccumulator.centroidMin_.x,
                                        boundsAccumulator.centroidMax_.y - boundsAccumulator.centroidMin_.y,
                                        boundsAccumulator.centroidMax_.z - boundsAccumulator.centroidMin_.z};
        int axis = 0;
        if (centroidExtent.y > centroidExtent.x && centroidExtent.y >= centroidExtent.z)
        {
            axis = 1;
        }
        else if (centroidExtent.z > centroidExtent.x && centroidExtent.z >= centroidExtent.y)
        {
            axis = 2;
        }

        if (centroidExtent.x <= 1e-6f && centroidExtent.y <= 1e-6f && centroidExtent.z <= 1e-6f)
        {
            // All centroids identical -> leaf
#ifdef BVH_USE_16_BITS_NODE
#ifdef BVH_PACK_NODE
            node.boundMinLeftRight_.w = uintAsFloat(Pack16x2(0, 0));
            node.boundMaxFirstCount_.w = uintAsFloat(Pack16x2(static_cast<uint16_t>(start), static_cast<uint16_t>(count)));
#else
            node.left_  = 0;
            node.right_ = 0;
            node.first_ = static_cast<uint16_t>(start);
            node.count_ = static_cast<uint16_t>(count);
#endif
#else
            node.left_  = -1;
            node.right_ = -1;
            node.first_ = start;
            node.count_ = count;
#endif
            return nodeIdx;
        }

        // SAH binning on chosen axis
        const float centroidMin = (axis == 0 ? boundsAccumulator.centroidMin_.x : (axis == 1 ? boundsAccumulator.centroidMin_.y : boundsAccumulator.centroidMin_.z));
        const float centroidMax = (axis == 0 ? boundsAccumulator.centroidMax_.x : (axis == 1 ? boundsAccumulator.centroidMax_.y : boundsAccumulator.centroidMax_.z));
        const float invW = (centroidMax > centroidMin) ? static_cast<float>(binCount) / (centroidMax - centroidMin) : 0.0f;

        std::vector<BVHBinData> bins(binCount);
        for (int b = 0; b < binCount; ++b)
        {
            bins[b].reset();
        }
        for (int i = start; i < end; ++i)
        {
            const PrimitiveProxy& p = proxy[result.objectIndices_[i]];
            const float key = (axis == 0 ? p.center_.x : (axis == 1 ? p.center_.y : p.center_.z));
            int b = invW > 0.0f ? static_cast<int>((key - centroidMin) * invW) : 0;
            b = (b < 0) ? 0 : (b >= binCount ? binCount - 1 : b);
            bins[b].add(p.boundMin_, p.boundMax_);
        }

        std::vector<float> leftArea(binCount);
        std::vector<float> rightArea(binCount);
        std::vector<int> leftCount(binCount);
        std::vector<int> rightCount(binCount);

        // left prefix
        float3 runMin = { FLT_MAX,  FLT_MAX,  FLT_MAX};
        float3 runMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        int runCount = 0;
        for (int b = 0; b < binCount; ++b)
        {
            if (bins[b].count_ > 0)
            {
                runMin = AABBMin(runMin, bins[b].boundMin_);
                runMax = AABBMax(runMax, bins[b].boundMax_);
            }
            runCount += bins[b].count_;
            leftArea[b] = (runCount > 0) ? AABBArea(runMin, runMax) : 0.0f;
            leftCount[b] = runCount;
        }

        // right suffix
        runMin = { FLT_MAX,  FLT_MAX,  FLT_MAX};
        runMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        runCount = 0;
        for (int b = binCount - 1; b >= 0; --b)
        {
            if (bins[b].count_ > 0)
            {
                runMin = AABBMin(runMin, bins[b].boundMin_);
                runMax = AABBMax(runMax, bins[b].boundMax_);
            }
            runCount += bins[b].count_;
            rightArea[b] = (runCount > 0) ? AABBArea(runMin, runMax) : 0.0f;
            rightCount[b] = runCount;
        }
        // SAH evaluation
#ifdef BVH_PACK_NODE
        const float parentArea = AABBArea(xyz(node.boundMinLeftRight_),xyz(node.boundMaxFirstCount_));
#else
        const float parentArea = AABBArea(node.boundMin_, node.boundMax_);
#endif
        constexpr float costI = 1.0f; // node traversal cost
        constexpr float costT = 1.0f; // primitive intersection cost
        float bestCost = FLT_MAX;
        int bestSplit = -1;
        for (int b = 0; b < binCount - 1; ++b)
        {
            const int nl = leftCount[b];
            const int nr = rightCount[b + 1];
            if (nl == 0 || nr == 0)
            {
                continue;
            }
            const float sah = costI +
                (leftArea[b] * static_cast<float>(nl) + rightArea[b + 1] * static_cast<float>(nr)) / fmaxf(parentArea, 1e-20f) * costT;
            if (sah < bestCost)
            {
                bestCost = sah;
                bestSplit = b;
            }
        }

        int mid = -1;
        if (bestSplit >= 0)
        {
            // Partition by bin id
            const int splitBin = bestSplit;
            auto itMid = std::partition(result.objectIndices_.begin() + start, result.objectIndices_.begin() + end,
                [&](int idx)
                {
                    const float key = (axis == 0 ? proxy[idx].center_.x :
                                      (axis == 1 ? proxy[idx].center_.y : proxy[idx].center_.z));
                    int b = invW > 0.0f ? static_cast<int>((key - centroidMin) * invW) : 0;
                    b = (b < 0) ? 0 : (b >= binCount ? binCount - 1 : b);
                    return b <= splitBin;
                });
            mid = static_cast<int>(itMid - result.objectIndices_.begin());
            if (mid == start || mid == end)
            {
                bestSplit = -1; // degenerate
            }
        }

        if (bestSplit < 0)
        {
            // Fallback: median split on centroid
            const int m = start + (count / 2);
            std::nth_element(result.objectIndices_.begin() + start, result.objectIndices_.begin() + m, result.objectIndices_.begin() + end,
                [&](int a, int b)
                {
                    const float ka = (axis == 0 ? proxy[a].center_.x :
                                     (axis == 1 ? proxy[a].center_.y : proxy[a].center_.z));
                    const float kb = (axis == 0 ? proxy[b].center_.x :
                                     (axis == 1 ? proxy[b].center_.y : proxy[b].center_.z));
                    return ka < kb;
                });
            mid = m;
            if (mid == start || mid == end)
            {
                // Ultimate fallback: leaf to avoid infinite recursion
#ifdef BVH_USE_16_BITS_NODE
#ifdef BVH_PACK_NODE
                node.boundMinLeftRight_.w = uintAsFloat(Pack16x2(0, 0));
                node.boundMaxFirstCount_.w = uintAsFloat(Pack16x2(static_cast<uint16_t>(start), static_cast<uint16_t>(count)));
#else
                node.left_  = 0;
                node.right_ = 0;
                node.first_ = static_cast<uint16_t>(start);
                node.count_ = static_cast<uint16_t>(count);
#endif
#else
                node.left_  = -1;
                node.right_ = -1;
                node.first_ = start;
                node.count_ = count;
#endif
                return nodeIdx;
            }
        }

        // Build children
        const int leftIdx  = build(start, mid, depth + 1);
        const int rightIdx = build(mid,   end, depth + 1);

        // Fill interior node
#ifdef BVH_USE_16_BITS_NODE
#ifdef BVH_PACK_NODE
        node.boundMinLeftRight_.w = uintAsFloat(Pack16x2(static_cast<uint16_t>(leftIdx), static_cast<uint16_t>(rightIdx)));
        node.boundMaxFirstCount_.w = uintAsFloat(Pack16x2(0, 0));
#else
        node.left_  = static_cast<uint16_t>(leftIdx);
        node.right_ = static_cast<uint16_t>(rightIdx);
        node.first_ = 0;
        node.count_ = 0;
#endif
#else
        node.left_  = leftIdx;
        node.right_ = rightIdx;
        node.first_ = -1;
        node.count_ = 0;
#endif
        // node.boundMin_/Max_ already set as union of this range
        return nodeIdx;
    };

    // root = 0 by construction
    (void)build(0, primitiveCount, 0);
    return result;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE bool FastAABBHit(
    const float3 rayOrigin, const float3 invDirection,
    const float3 boundMin, const float3 boundMax,
    const float tMin, const float tMax)
{
    float t0 = (boundMin.x - rayOrigin.x) * invDirection.x;
    float t1 = (boundMax.x - rayOrigin.x) * invDirection.x;
    float tMin2 = fminf(t0, t1);
    float tMax2 = fmaxf(t0, t1);
    t0 = (boundMin.y - rayOrigin.y) * invDirection.y;
    t1 = (boundMax.y - rayOrigin.y) * invDirection.y;
    tMin2 = fmaxf(tMin2, fminf(t0, t1));
    tMax2 = fminf(tMax2, fmaxf(t0, t1));
    t0 = (boundMin.z - rayOrigin.z) * invDirection.z;
    t1 = (boundMax.z - rayOrigin.z) * invDirection.z;
    tMin2 = fmaxf(tMin2, fminf(t0, t1));
    tMax2 = fminf(tMax2, fmaxf(t0, t1));
    return tMax2 >= fmaxf(0.0f, tMin) && tMin2 <= tMax;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE bool FastAABBHitWithEnter(
    const float3 rayOrigin, const float3 invDirection,
    const float3 boundMin, const float3 boundMax,
    const float tMin, const float tMax,
    float& tEnterOut)
{
    const float tx0 = (boundMin.x - rayOrigin.x) * invDirection.x;
    const float tx1 = (boundMax.x - rayOrigin.x) * invDirection.x;
    float tEnter = fminf(tx0, tx1);
    float tExit  = fmaxf(tx0, tx1);

    const float ty0 = (boundMin.y - rayOrigin.y) * invDirection.y;
    const float ty1 = (boundMax.y - rayOrigin.y) * invDirection.y;
    tEnter = fmaxf(tEnter, fminf(ty0, ty1));
    tExit  = fminf(tExit,  fmaxf(ty0, ty1));

    float tz0 = (boundMin.z - rayOrigin.z) * invDirection.z;
    float tz1 = (boundMax.z - rayOrigin.z) * invDirection.z;
    tEnter = fmaxf(tEnter, fminf(tz0, tz1));
    tExit  = fminf(tExit,  fmaxf(tz0, tz1));
    if (tExit >= fmaxf(0.0f, tMin) && tEnter <= tMax)
    {
        tEnterOut = tEnter;
        return true;
    }
    return false;
}