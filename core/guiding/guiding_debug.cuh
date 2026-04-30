// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "guiding_bin.cuh"
#include "render/color.cuh"

FUNCTION_MODIFIER_DEVICE_INLINE uint32_t WangHash(uint32_t x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u; x = x ^ (x >> 4);
    x *= 0x27d4eb2du; x = x ^ (x >> 15);
    return x;
}

FUNCTION_MODIFIER_DEVICE_INLINE float WangHash01(uint32_t seed)
{
    return static_cast<float>(WangHash(seed)) * (1.0f/4294967296.0f);
};

    
FUNCTION_MODIFIER_DEVICE_INLINE int3 GetVoxelFaceStats(const int ix, const int iy, const int iz, const int3 dim, const size_t* indirectVolume)
{
    const size_t base = pg::Flatten3D(ix,iy,iz,dim) * PATH_GUIDING_FACE_COUNT;
    int neg1 = 0;
    size_t firstAssignIndex = SIZE_T_MAX;
#pragma unroll
    for (int f=0; f<PATH_GUIDING_FACE_COUNT; ++f)
    {
        const size_t idx = indirectVolume[base + f];
        if (idx == SIZE_T_MAX)
        {
            ++neg1;
        }
        else
        {
            firstAssignIndex = min(idx, firstAssignIndex);
        }
    }
    const int isEmpty = (neg1 == PATH_GUIDING_FACE_COUNT) ? 1 : 0;
    return make_int3(isEmpty, neg1, static_cast<uint32_t>(firstAssignIndex ^ (firstAssignIndex >> 32)));
};

FUNCTION_MODIFIER_DEVICE_INLINE float4 TracePathGuidingIndirectVolume(float3 rayOrigin, float3 rayDirection, const size_t* indirectVolume, const int3 volumeStart, const int3 volumeEnd)
{
    if (indirectVolume == nullptr)
    {
        return make_float4(0,0,0,0);
    }

    const int3 vMin = volumeStart;
    const int3 vMax = volumeEnd;
    const int3 dim  = pg::DimsFromBoundsInclusive(vMin, vMax);
    if (dim.x<=0 || dim.y<=0 || dim.z<=0)
    {
        return make_float4(0,0,0,0);
    }
    const float3 boxMinWS = make_float3(vMin.x * PATH_GUIDING_RESOLUTION_UNIT, vMin.y * PATH_GUIDING_RESOLUTION_UNIT, vMin.z * PATH_GUIDING_RESOLUTION_UNIT);
    const float3 boxMaxWS = make_float3((vMax.x + 1) * PATH_GUIDING_RESOLUTION_UNIT, (vMax.y + 1) * PATH_GUIDING_RESOLUTION_UNIT, (vMax.z + 1) * PATH_GUIDING_RESOLUTION_UNIT);

    const float2 tmm = object::IntersectAABBBase(rayOrigin, rayDirection, boxMinWS, boxMaxWS);
    const float tEnter = tmm.x;
    const float tExit  = tmm.y;
    if (tExit < tEnter || tExit < 0.0f)
    {
        return make_float4(0,0,0,0);
    }
    float t = fmaxf(tEnter, 0.0f);
    float3 pos = rayOrigin + t * rayDirection;

    float3 rel = (pos - boxMinWS) * (1.0f / PATH_GUIDING_RESOLUTION_UNIT);
    int ix = max(0, min(static_cast<int>(floorf(rel.x)), dim.x-1));
    int iy = max(0, min(static_cast<int>(floorf(rel.y)), dim.y-1));
    int iz = max(0, min(static_cast<int>(floorf(rel.z)), dim.z-1));

    const int stepX = (rayDirection.x > 0.0f) ? 1 : -1;
    const int stepY = (rayDirection.y > 0.0f) ? 1 : -1;
    const int stepZ = (rayDirection.z > 0.0f) ? 1 : -1;

    constexpr float infV = 1e30f;
    float txDelta = (rayDirection.x==0.0f) ? infV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.x);
    float tyDelta = (rayDirection.y==0.0f) ? infV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.y);
    float tzDelta = (rayDirection.z==0.0f) ? infV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.z);

    float nextBx = boxMinWS.x + ((stepX>0)? (ix+1) : ix) * PATH_GUIDING_RESOLUTION_UNIT;
    float nextBy = boxMinWS.y + ((stepY>0)? (iy+1) : iy) * PATH_GUIDING_RESOLUTION_UNIT;
    float nextBz = boxMinWS.z + ((stepZ>0)? (iz+1) : iz) * PATH_GUIDING_RESOLUTION_UNIT;

    float txMax = (rayDirection.x==0.0f) ? infV : (nextBx - pos.x) / rayDirection.x;
    float tyMax = (rayDirection.y==0.0f) ? infV : (nextBy - pos.y) / rayDirection.y;
    float tzMax = (rayDirection.z==0.0f) ? infV : (nextBz - pos.z) / rayDirection.z;

    float3 accumRGB = make_float3(0,0,0);
    float accumA = 0.0f;

    constexpr float baseAlpha = 0.99f;
    const int maxIters = dim.x + dim.y + dim.z + 4;

    for (int it=0; it<maxIters && t <= tExit && ix>=0 && iy>=0 && iz>=0 && ix<dim.x && iy<dim.y && iz<dim.z; ++it)
    {
        const int3 grid = pg::GetVoxelLocation(pos);
        const int3 stat = GetVoxelFaceStats(ix, iy, iz, dim, indirectVolume);
        const bool isEmpty = (stat.x == 1);
        const int neg1Count = stat.y;
        const int seedIdx = stat.z;

        float alpha = baseAlpha * (isEmpty ? 0.1f : 1.0f);
        float3 rgb = make_float3(0,0,0);
        if (!isEmpty)
        {
            float saturate = fminf(fmaxf(static_cast<float>(neg1Count) / static_cast<float>(PATH_GUIDING_FACE_COUNT), 0.0f), 1.0f) * 0.5f + 0.5f;
            float value = isEmpty ? 0.0f : 1.0f;
            uint32_t seed = (seedIdx != -1) ? static_cast<uint32_t>(seedIdx) : (ix*73856093u ^ iy*19349663u ^ iz*83492791u);
            float hue = WangHash01(seed);
            rgb = color::HSV2RGB(hue, saturate, value);
        }
        float oneMinusA = (1.0f - accumA);
        accumRGB.x += rgb.x * alpha * oneMinusA;
        accumRGB.y += rgb.y * alpha * oneMinusA;
        accumRGB.z += rgb.z * alpha * oneMinusA;
        accumA += alpha * oneMinusA;
        if (accumA > 0.995f)
        {
            break;
        }
        if (txMax < tyMax && txMax < tzMax)
        {
            const float adv = txMax;
            t += adv;
            pos = pos + adv * rayDirection;
            txMax = txDelta;
            tyMax -= adv;
            tzMax -= adv;
            ix += stepX;
        }
        else if (tyMax < tzMax)
        {
            const float adv = tyMax;
            t += adv;
            pos = pos + adv * rayDirection;
            tyMax = tyDelta;
            txMax -= adv;
            tzMax -= adv;
            iy += stepY;
        }
        else
        {
            const float adv = tzMax;
            t += adv;
            pos = pos + adv * rayDirection;
            tzMax = tzDelta;
            txMax -= adv;
            tyMax -= adv;
            iz += stepZ;
        }
    }
    return {accumRGB.x, accumRGB.y, accumRGB.z, accumA};
}

#ifdef USE_3_CHANNEL_PROBE
FUNCTION_MODIFIER_DEVICE_INLINE float3 SampleOctahedronLeafNearest(const Octahedron& probe, const float3 dirN)
#else
FUNCTION_MODIFIER_DEVICE_INLINE float SampleOctahedronLeafNearest(const Octahedron& probe, const float3 dirN)
#endif
{
    constexpr int res = Octahedron::resolution_;
    const float2 uv = pg::SphereToUV(dirN); // [0,1]
    int ix = static_cast<int>(floorf(uv.x * res));
    int iy = static_cast<int>(floorf(uv.y * res));
    ix = max(0, min(res - 1, ix));
    iy = max(0, min(res - 1, iy));
#ifdef USE_3_CHANNEL_PROBE
    return xyz(probe.Texel(ix, iy, 0)); //  / luminance(probe.Texel(0, 0, Octahedron::maxMipmapLevel_ - 1))
#else
    return probe.Texel(ix, iy, 0) / probe.Texel(0, 0, Octahedron::maxMipmapLevel_ - 1);
#endif
}

FUNCTION_MODIFIER_DEVICE_INLINE float4 TracePathGuidingIndirectVolumeDetailed(float3 rayOrigin, float3 rayDirection, const size_t* indirectVolume, const int3 volumeStart, const int3 volumeEnd, const Octahedron* probes)
{
    if (!indirectVolume)
    {
        return make_float4(0,0,0,0);
    }

    auto GetProbeIndexAtFace = [&] __device__ (int ix, int iy, int iz, int face, int3 dim)->size_t
    {
        const size_t base = pg::Flatten3D(ix, iy, iz, dim) * PATH_GUIDING_FACE_COUNT;
        return indirectVolume[base + face];
    };

    const int3 vMin = volumeStart, vMax = volumeEnd;
    const int3 dim = pg::DimsFromBoundsInclusive(vMin, vMax);
    if (dim.x<=0 || dim.y<=0 || dim.z<=0)
    {
        return make_float4(0,0,0,0);
    }
    const float3 boxMinWS = make_float3(
        static_cast<float>(vMin.x) * PATH_GUIDING_RESOLUTION_UNIT,
        static_cast<float>(vMin.y) * PATH_GUIDING_RESOLUTION_UNIT,
        static_cast<float>(vMin.z) * PATH_GUIDING_RESOLUTION_UNIT
        );
    const float3 boxMaxWS = make_float3(
        static_cast<float>(vMax.x+1) * PATH_GUIDING_RESOLUTION_UNIT,
        static_cast<float>(vMax.y+1) * PATH_GUIDING_RESOLUTION_UNIT,
        static_cast<float>(vMax.z+1) * PATH_GUIDING_RESOLUTION_UNIT
        );

    const float2 tmm = object::IntersectAABBBase(rayOrigin, rayDirection, boxMinWS, boxMaxWS);
    float tEnter = tmm.x, tExit = tmm.y;
    if (tExit < tEnter || tExit < 0.0f)
    {
        return make_float4(0,0,0,0);
    }
    float t = fmaxf(tEnter, 0.0f);
    float3 pos = rayDirection * t + rayOrigin;

    float3 rel = (pos - boxMinWS) * (1.0f / PATH_GUIDING_RESOLUTION_UNIT);
    int ix = max(0, min(floorToInt(rel.x), dim.x-1));
    int iy = max(0, min(floorToInt(rel.y), dim.y-1));
    int iz = max(0, min(floorToInt(rel.z), dim.z-1));

    const int stepX = (rayDirection.x > 0.0f) ? 1 : -1;
    const int stepY = (rayDirection.y > 0.0f) ? 1 : -1;
    const int stepZ = (rayDirection.z > 0.0f) ? 1 : -1;

    constexpr float INFV = 1e30f;
    float txDelta = (rayDirection.x==0.0f) ? INFV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.x);
    float tyDelta = (rayDirection.y==0.0f) ? INFV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.y);
    float tzDelta = (rayDirection.z==0.0f) ? INFV : PATH_GUIDING_RESOLUTION_UNIT / fabsf(rayDirection.z);

    float nextBx = boxMinWS.x + static_cast<float>((stepX>0)? (ix + 1) : ix) * PATH_GUIDING_RESOLUTION_UNIT;
    float nextBy = boxMinWS.y + static_cast<float>((stepY>0)? (iy + 1) : iy) * PATH_GUIDING_RESOLUTION_UNIT;
    float nextBz = boxMinWS.z + static_cast<float>((stepZ>0)? (iz + 1) : iz) * PATH_GUIDING_RESOLUTION_UNIT;

    float txMax = (rayDirection.x==0.0f) ? INFV : (nextBx - pos.x) / rayDirection.x;
    float tyMax = (rayDirection.y==0.0f) ? INFV : (nextBy - pos.y) / rayDirection.y;
    float tzMax = (rayDirection.z==0.0f) ? INFV : (nextBz - pos.z) / rayDirection.z;

    const int maxIters = dim.x + dim.y + dim.z + 4;

    const float rdLen = sqrtf(rayDirection.x*rayDirection.x + rayDirection.y*rayDirection.y + rayDirection.z*rayDirection.z);
    float3 rdN = (rdLen > 0.0f) ? make_float3(rayDirection.x/rdLen, rayDirection.y/rdLen, rayDirection.z/rdLen)
                                : make_float3(0,0,0);

    for (int it=0; it<maxIters && t<=tExit && ix>=0 && iy>=0 && iz>=0 && ix<dim.x && iy<dim.y && iz<dim.z; ++it)
    {
        float advNext; int stepAxis;
        if (txMax <= tyMax && txMax <= tzMax)
        {
            advNext = txMax;
            stepAxis = 0;
        }
        else if (tyMax <= tzMax)
        {
            advNext = tyMax;
            stepAxis = 1;
        }
        else
        {
            advNext = tzMax;
            stepAxis = 2;
        }
        const float3 cellMin = make_float3(
            boxMinWS.x + static_cast<float>(ix) * PATH_GUIDING_RESOLUTION_UNIT,
            boxMinWS.y + static_cast<float>(iy) * PATH_GUIDING_RESOLUTION_UNIT,
            boxMinWS.z + static_cast<float>(iz) * PATH_GUIDING_RESOLUTION_UNIT
            );
        const float3 cellMax = make_float3(
            cellMin.x + PATH_GUIDING_RESOLUTION_UNIT,
            cellMin.y + PATH_GUIDING_RESOLUTION_UNIT,
            cellMin.z + PATH_GUIDING_RESOLUTION_UNIT
            );
        const float3 center = make_float3(0.5f*(cellMin.x+cellMax.x),
                                          0.5f*(cellMin.y+cellMax.y),
                                          0.5f*(cellMin.z+cellMax.z));
        constexpr float centerBoxScale = 0.5f;
        constexpr float centerBoxReScale = 0.5f;
        constexpr float sphereRadius = (1.0f - centerBoxScale) * 0.5f * 0.5f;
        constexpr float he = centerBoxReScale * centerBoxScale * 0.5f * PATH_GUIDING_RESOLUTION_UNIT;
        const float3 subMin = make_float3(center.x - he, center.y - he, center.z - he);
        const float3 subMax = make_float3(center.x + he, center.y + he, center.z + he);

        constexpr float sphOffset = (centerBoxScale * 0.5 + sphereRadius) * PATH_GUIDING_RESOLUTION_UNIT;
        constexpr float sphR = sphereRadius * PATH_GUIDING_RESOLUTION_UNIT;

        const int3 stat = GetVoxelFaceStats(ix, iy, iz, dim, indirectVolume);
        const bool voxelEmpty = (stat.x == 1);
        const int  neg1Count  = stat.y;
        const uint32_t seedFold = stat.z;

        if (!voxelEmpty)
        {
            const float2 tBox = object::IntersectAABBBase(rayOrigin, rayDirection, subMin, subMax);
            const float tBoxEnter = fmaxf(tBox.x, t);
            const bool  hitBox    = (tBoxEnter <= tBox.y) && (tBoxEnter <= t + advNext);

            float tSphBest = INFV;
            int tSphBestFace = -1;
            [[maybe_unused]] float tSphBestN = INFV;
            float3 sphCenterBest = make_float3(0,0,0);

            if (rdLen > 0.0f)
            {
                CUDA_UNROLL
                for (int f=0; f<PATH_GUIDING_FACE_COUNT; ++f)
                {
                    const float3 nf = pg::FaceDirectionFromIndex(f);
                    const float3 centerF = make_float3(
                        center.x + nf.x * sphOffset,
                        center.y + nf.y * sphOffset,
                        center.z + nf.z * sphOffset
                        );
                    float3 roLocal = make_float3(
                        rayOrigin.x - centerF.x,
                        rayOrigin.y - centerF.y,
                        rayOrigin.z - centerF.z
                        );
                    bool inside = false;
                    float tN = object::IntersectSphereBase(roLocal, rdN, sphR, inside);
                    if (tN >= 0.0f && tN < INFV)
                    {
                        float tParam = tN / rdLen;
                        if (tParam >= t && tParam <= t + advNext && tParam < tSphBest)
                        {
                            tSphBest = tParam;
                            tSphBestFace = f;
                            tSphBestN = tN;
                            sphCenterBest = centerF;
                        }
                    }
                }
            }

            bool pickBox = false;
            if (hitBox && tSphBest < INFV)
            {
                pickBox = (tBoxEnter <= tSphBest);
            }
            else if (hitBox)
            {
                pickBox = true;
            }
            if (pickBox)
            {
                float saturate = fminf(fmaxf(static_cast<float>(neg1Count)/static_cast<float>(PATH_GUIDING_FACE_COUNT), 0.0f), 1.0f) * 0.5f + 0.5f;
                float value = 1.0f;
                float hue = WangHash01(seedFold);
                float3 rgb = color::HSV2RGB(hue, saturate, value);
                return make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
            }
            else if (tSphBest < INFV && tSphBestFace >= 0)
            {
                const int faceIdx = tSphBestFace;
                const size_t probeIndex = GetProbeIndexAtFace(ix, iy, iz, faceIdx, dim);
                if (probeIndex != SIZE_T_MAX)
                {
                    const Octahedron& probe = probes[probeIndex];
                    const float3 hitWS = make_float3(
                        rayOrigin.x + rayDirection.x * tSphBest,
                        rayOrigin.y + rayDirection.y * tSphBest,
                        rayOrigin.z + rayDirection.z * tSphBest);
                    float3 n = make_float3(
                        hitWS.x - sphCenterBest.x,
                        hitWS.y - sphCenterBest.y,
                        hitWS.z - sphCenterBest.z);
                    const float lenN = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
                    if (lenN > 0.0f)
                    {
                        n.x /= lenN; n.y /= lenN; n.z /= lenN;
                    } else
                    {
                        n = pg::FaceDirectionFromIndex(faceIdx);
                    }
#ifdef USE_3_CHANNEL_PROBE
#ifdef USE_SPECTRUM_RENDERING
                    const float3 rgb = XYZ2SRGBLinearD65(SampleOctahedronLeafNearest(probe, n));
#else
                    const float3 rgb = SampleOctahedronLeafNearest(probe, n);
                    //const float value = luminance(SampleOctahedronLeafNearest(probe, n));
                    //const float3 rgb = Color::ProbeValueToRGB(value, Color::EHeatMap::VIRIDIS);
#endif
#else
                    const float value = SampleOctahedronLeafNearest(probe, n);
                    const float3 rgb = Color::ProbeValueToRGB(value, Color::EHeatMap::VIRIDIS);
#endif
                    return make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
                }
                else
                {
#ifdef DEBUG_PATH_GUIDING_INDIRECT_VOLUME_CULL_INVALID_DIRECTION
                    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#else
                    return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
#endif
                }
            }
        }
        const float adv = advNext;
        t += adv;
        pos = rayDirection * adv + pos;
        if(stepAxis == 0)
        {
            txMax = txDelta;
            tyMax -= adv;
            tzMax -= adv;
            ix += stepX;
        }
        else if (stepAxis == 1)
        {
            tyMax = tyDelta;
            txMax -= adv;
            tzMax -= adv;
            iy += stepY;
        }
        else
        {
            tzMax = tzDelta;
            txMax -= adv;
            tyMax -= adv;
            iz += stepZ;
        }
    }
    return make_float4(0,0,0,0);
}