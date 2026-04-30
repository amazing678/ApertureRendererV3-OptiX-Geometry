// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "random.cuh"
#include "render/sample.cuh"
#include "octahedron.cuh"
#include <cuda_runtime.h>

namespace pg
{
    [[nodiscard]] FUNCTION_MODIFIER_INLINE int Get6FaceIndex(const float3 n)
    {
        const float3 a = abs(n);
        if (a.x >= a.y && a.x >= a.z)
        {
            return n.x >= 0.0f ? 0 : 1;
        }
        else if (a.y >= a.x && a.y >= a.z)
        {
            return n.y >= 0.0f ? 2 : 3;
        }
        else
        {
            return n.z >= 0.0f ? 4 : 5;
        }
    }
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 FaceNormal6FromIndex(const int f)
    {
        switch (f)
        {
        case 0: return make_float3(+1.0f, 0.0f, 0.0f);
        case 1: return make_float3(-1.0f, 0.0f, 0.0f);
        case 2: return make_float3(0.0f, +1.0f, 0.0f);
        case 3: return make_float3(0.0f, -1.0f, 0.0f);
        case 4: return make_float3(0.0f, 0.0f, +1.0f);
        default:return make_float3(0.0f, 0.0f, -1.0f);
        }
    }
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 FaceNormal20FromIndex(const int f)
    {
        switch (f)
        {
        case  0: return make_float3(-0.577350269f,  0.577350269f,  0.577350269f);
        case  1: return make_float3( 0.000000000f,  0.934172359f,  0.356822090f);
        case  2: return make_float3( 0.000000000f,  0.934172359f, -0.356822090f);
        case  3: return make_float3(-0.577350269f,  0.577350269f, -0.577350269f);
        case  4: return make_float3(-0.934172359f,  0.356822090f,  0.000000000f);
        case  5: return make_float3( 0.577350269f,  0.577350269f,  0.577350269f);
        case  6: return make_float3(-0.356822090f,  0.000000000f,  0.934172359f);
        case  7: return make_float3(-0.934172359f, -0.356822090f,  0.000000000f);
        case  8: return make_float3(-0.356822090f,  0.000000000f, -0.934172359f);
        case  9: return make_float3( 0.577350269f,  0.577350269f, -0.577350269f);
        case 10: return make_float3( 0.577350269f, -0.577350269f,  0.577350269f);
        case 11: return make_float3( 0.000000000f, -0.934172359f,  0.356822090f);
        case 12: return make_float3( 0.000000000f, -0.934172359f, -0.356822090f);
        case 13: return make_float3( 0.577350269f, -0.577350269f, -0.577350269f);
        case 14: return make_float3( 0.934172359f, -0.356822090f,  0.000000000f);
        case 15: return make_float3( 0.356822090f,  0.000000000f,  0.934172359f);
        case 16: return make_float3(-0.577350269f, -0.577350269f,  0.577350269f);
        case 17: return make_float3(-0.577350269f, -0.577350269f, -0.577350269f);
        case 18: return make_float3( 0.356822090f,  0.000000000f, -0.934172359f);
        case 19: return make_float3( 0.934172359f,  0.356822090f,  0.000000000f);
        default: return make_float3(0.0f, 0.0f, 1.0f);
        }
    }
    [[nodiscard]] FUNCTION_MODIFIER_INLINE int Get20FaceIndex(const float3 n)
    {
        const float3 d = normalize(n);
        int best = 0;
        float bestDot = -1e30f;
        CUDA_UNROLL
        for (int f = 0; f < 20; ++f)
        {
            const float3 c = FaceNormal20FromIndex(f);
            const float v = d.x * c.x + d.y * c.y + d.z * c.z;
            if (v > bestDot)
            {
                bestDot = v;
                best = f;
            }
        }
        return best;
    }
        
    [[nodiscard]] FUNCTION_MODIFIER_INLINE int GetFaceIndex(const float3 direction) // pair with FaceNormalFromIndex
    {
#ifdef PATH_GUIDING_ICOSAHEDRON
        return Get20FaceIndex(direction);
#else
        return Get6FaceIndex(direction);
#endif
    }
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float3 FaceDirectionFromIndex(const int faceIndex) // pair with GetFaceIndex
    {
#ifdef PATH_GUIDING_ICOSAHEDRON
        return FaceNormal20FromIndex(faceIndex);
#else
        return FaceNormal6FromIndex(faceIndex);
#endif
    }
    [[nodiscard]] FUNCTION_MODIFIER_INLINE int3 GetVoxelLocation(const float3 location)
    {
        constexpr float invUnit = 1.0f / PATH_GUIDING_RESOLUTION_UNIT;
        const float3 scaled = location * invUnit;
        return floorToInt(scaled);
    }
        
    [[nodiscard]] FUNCTION_MODIFIER_INLINE int4 GetVoxelGrid(const float3 location, const float3 direction) // IN WORLD SPACE!
    {
        const int3 grid = GetVoxelLocation(location);
        const int faceIndex = GetFaceIndex(direction);
        return {grid.x, grid.y, grid.z, faceIndex};
    }
}