// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#include <curand_kernel.h>

#include "vector.cuh"
#include "random.cuh"
#include "render/object.cuh"

struct LightSample 
{
    // int lightIndex_;
    int objectIndex_;
    int facetIndex_;
    float3 samplePoint_;
    float3 samplePointNormal_;
    float3 sampleEmissive_;
    
    float pFacet_; // pLight * pLightFacet
    float pA_; // surface prob
    
    bool bIsValid_ = false; // is valid
};

FUNCTION_MODIFIER_INLINE
void GetOBBAxes(const SceneObject& o, float3& ux, float3& uy, float3& uz)
{
    const float3x3& M = o.objectToWorld_;
    ux = normalize(make_f3(M.x.x, M.y.x, M.z.x)); // local +X in world
    uy = normalize(make_f3(M.x.y, M.y.y, M.z.y)); // local +Y in world
    uz = normalize(make_f3(M.x.z, M.y.z, M.z.z)); // local +Z in world
}



FUNCTION_MODIFIER_INLINE float AreaToSolidAnglePDF(const float3 surfacePoint, const float3 lightPoint, const float3 lightPointNormal, float pATotal)
{
    const float3 surfaceToLight = lightPoint - surfacePoint;
    const float distance2 = fmaxf(dot(surfaceToLight, surfaceToLight), 1e-20f);
    const float3 surfaceToLightDirection = surfaceToLight * rsqrtf(distance2);
    const float cosY= fmaxf(0.0f, dot(lightPointNormal, -surfaceToLightDirection));
    if (cosY <= 0.0f)
    {
        return 0.0f;
    }
    return pATotal * (distance2 / cosY);
}

FUNCTION_MODIFIER_INLINE
float NEELightPDFOmegaFromArea(const float3 surfacePoint, const LightSample& lightSample)
{
    const float pATotal = lightSample.pFacet_ * lightSample.pA_;
    return AreaToSolidAnglePDF(surfacePoint, lightSample.samplePoint_, lightSample.samplePointNormal_, pATotal);
}

FUNCTION_MODIFIER_DEVICE_INLINE
float NEEFacingWeight_Box(const float3 facetNormal, const float3 facetCenter, const float3 surfacePoint, float area)
{
    const float3 lightToSurface = normalize(surfacePoint - facetCenter);
    const float cos = fmaxf(0.0f, dot(facetNormal, lightToSurface));
    return area * cos;
}

// must PAIR with NEESampleBoxAreaFacing
FUNCTION_MODIFIER_DEVICE_INLINE float NEEPDFOmegaFromHit_Box(const SceneObject& lightObject, const AdditionalLightInfo& lightInfo, const float3 surfacePoint, const float3 lightPoint, const float3 lightNormal)
{
    float3 uX,uY,uZ;
    GetOBBAxes(lightObject, uX, uY, uZ);
    
    const float3 center = lightObject.center_;
    const float3 extent = lightObject.extent_;
    const float3 facetNormals[6] = { uX, -uX,  uY, -uY,  uZ, -uZ };
    const float3 facetCenters[6] =
    {
        center + uX * extent.x, center - uX * extent.x,
        center + uY * extent.y, center - uY * extent.y,
        center + uZ * extent.z, center - uZ * extent.z
    };
    const float side1[6] = { extent.y, extent.y, extent.x, extent.x, extent.x, extent.x };
    const float side2[6] = { extent.z, extent.z, extent.z, extent.z, extent.y, extent.y };
    
    float area[6];
#pragma unroll
    for (int f=0; f<6; ++f)
    {
        area[f] = 4.0f * side1[f] * side2[f];
    }
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
    int hitFacet = -1;
    { // find hit facet
#pragma unroll
        for (int f=0; f<6; ++f)
        {
            if (dot(facetNormals[f], lightNormal) > 0.999f)
            {
                hitFacet = f;
                break;
            }
        }
        if (hitFacet < 0)
        {
            float best = 1e30f;
            for (int f=0; f<6; ++f)
            {
                const float d = fabsf(dot(facetNormals[f], lightPoint - facetCenters[f]));
                if (d < best)
                {
                    best = d;
                    hitFacet = f;
                }
            }
        }
    }
#endif
    float sumWeights = 0.0f;
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
    float weightHit = 0.0f;
    float areaHit = 0.0f;
#endif
#pragma unroll
    for (int f=0; f<6; ++f)  // NOLINT(modernize-loop-convert)
    {
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
        const float weight = NEEFacingWeight_Box(facetNormals[f], facetCenters[f], surfacePoint, area[f]);
        sumWeights += weight;
        if (f == hitFacet)
        {
            weightHit = weight;
            areaHit = area[f];
        }
#else
        if (dot(facetNormals[f], surfacePoint - facetCenters[f]) > 0.0f)
        {
            sumWeights += area[f];
        }
#endif
    }
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
    if (sumWeights <= 0.0f || weightHit <= 0.0f)
#else
    if (sumWeights <= 0.0f)
#endif
    {
        return 0.0f;
    }
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
    const float pATotal = lightInfo.selectProb_ * (1.0f / areaHit) * (weightHit / sumWeights);
#else
    const float pATotal = lightInfo.selectProb_  / sumWeights;
#endif
    return AreaToSolidAnglePDF(surfacePoint, lightPoint, lightNormal, pATotal);
}

// must PAIR with NEEPDFOmegaFromHit_Box
FUNCTION_MODIFIER_DEVICE_INLINE LightSample NEESampleBoxAreaFacing(curandState* seed, const SceneObject& box, const float3 currentPoint)
{
    LightSample lightSample = {};
    
    float3 uX;
    float3 uY;
    float3 uZ;
    GetOBBAxes(box, uX, uY, uZ);
    const float3 boxCenter = box.center_;
    const float3 boxExtent = box.extent_;
    // 6 center & normal (+/-X, +/-Y, +/-Z)
    const float3 facetNormals[6] = {uX, -uX,  uY, -uY,  uZ, -uZ};
    const float3 facetCenters[6] =
    {
        boxCenter + uX * boxExtent.x, boxCenter - uX * boxExtent.x,
        boxCenter + uY * boxExtent.y, boxCenter - uY * boxExtent.y,
        boxCenter + uZ * boxExtent.z, boxCenter - uZ * boxExtent.z,
    };
    // Tangent & Bitangent
    const float3 facetTangent[6] = {uY,  uY,  uX,  uX,  uX,  uX};
    const float3 facetBitangent[6] = {uZ,  uZ,  uZ,  uZ,  uY,  uY};
    // Area
    const float side1[6] = {boxExtent.y, boxExtent.y, boxExtent.x, boxExtent.x, boxExtent.x, boxExtent.x};
    const float side2[6] = {boxExtent.z, boxExtent.z, boxExtent.z, boxExtent.z, boxExtent.y, boxExtent.y};
    float area[6];
#pragma unroll
    for (int f=0; f<6; f++)
    {
        area[f] = 4.0f * side1[f] * side2[f];
    }
    // find facet face currentPoint
    int validFacetIndices[6];
    float validFacetWeights[6];
    int validFacetCount = 0;
    float validFacetWeightsSum = 0;
    
#pragma unroll
    for (int facetIndex = 0; facetIndex < 6; ++facetIndex)
    {
#ifdef NEE_USE_CUSTOM_FACET_WEIGHT
        const float currentWeight = NEEFacingWeight_Box(facetNormals[facetIndex], facetCenters[facetIndex], currentPoint, area[facetIndex]);
        if (currentWeight > 0.0f)
        {
            validFacetIndices[validFacetCount] = facetIndex;
            validFacetWeights[validFacetCount] = currentWeight;
            validFacetWeightsSum += currentWeight;
            ++validFacetCount;
        }
#else
        if (dot(facetNormals[facetIndex], currentPoint - facetCenters[facetIndex]) > 0.0f)
        {
            validFacetIndices[validFacetCount] = facetIndex;
            validFacetWeights[validFacetCount] = area[facetIndex];
            validFacetWeightsSum += area[facetIndex];
            ++validFacetCount;
        }
#endif
    }
    
    if (validFacetCount == 0)
    {
        return lightSample; // not valid
    }

    int pickedSlot = 0;
    int pickedFacetIndex = validFacetIndices[0];
    {
        const float r = Rand1(seed) * validFacetWeightsSum;
        float acc = 0;
        for (int k = 0; k < validFacetCount; k++)
        {
            acc += validFacetWeights[k];
            if (r <= acc)
            {
                pickedSlot = k;
                pickedFacetIndex = validFacetIndices[k];
                break;
            }
        }
    }
    
    const float u1 = 2.0f * Rand1(seed) - 1.0f;
    const float u2 = 2.0f * Rand1(seed) - 1.0f;
    const float3 samplePoint = facetCenters[pickedFacetIndex] +
        facetTangent[pickedFacetIndex] * (u1 * side1[pickedFacetIndex]) +
        facetBitangent[pickedFacetIndex] * (u2 * side2[pickedFacetIndex]);

    lightSample.facetIndex_ = pickedFacetIndex;
    lightSample.samplePoint_ = samplePoint;
    lightSample.samplePointNormal_ = facetNormals[pickedFacetIndex];
    
    lightSample.sampleEmissive_ = box.material_.texture_.baseTexture_.emissive_;
    
    lightSample.pFacet_ = 1.0f * (validFacetWeights[pickedSlot] / validFacetWeightsSum);
    lightSample.pA_ = 1.0f / area[pickedFacetIndex];
    lightSample.bIsValid_ = true;
    return lightSample;
}

// must PAIR with NEESampleSphereCapArea
FUNCTION_MODIFIER_DEVICE_INLINE float NEEPDFOmegaFromHit_Sphere(const SceneObject& lightObject, const AdditionalLightInfo& lightInfo, const float3 surfacePoint, const float3 lightPoint, const float3 lightNormal)
{
    const float3 sphereCenter = lightObject.center_;
    const float sphereRadius = max3(lightObject.extent_);
    const float distanceToSphere = length(surfacePoint - sphereCenter);
    if (distanceToSphere <= sphereRadius)
    {
        return 0.0f;
    }
    const float cosC = clamp(sphereRadius / distanceToSphere, 0.0f, 1.0f);
    const float areaCap = 2.0f * CUDART_PI_F * sphereRadius * sphereRadius * (1.0f - cosC); // must same asNEESampleSphereCapArea
    if (areaCap <= 1e-20f)
    {
        return 0.0f;
    }
    const float pATotal = lightInfo.selectProb_ / areaCap;
    return AreaToSolidAnglePDF(surfacePoint, lightPoint, lightNormal, pATotal);
}

// must PAIR with NEEPDFOmegaFromHit_Sphere
FUNCTION_MODIFIER_DEVICE_INLINE LightSample NEESampleSphereCapArea(curandState* seed, const SceneObject& sphere, const float3 currentPoint)
{
    LightSample lightSample{};
    const float3 sphereCenter = sphere.center_;
    const float sphereRadius = max3(sphere.extent_);
    const float3 pointToSphereCenter = currentPoint - sphereCenter;
    const float distanceToSphereCenter = length(pointToSphereCenter);
    if (distanceToSphereCenter <= sphereRadius)
    {
        return lightSample;
    }
    const float3 axis = pointToSphereCenter / distanceToSphereCenter; // Axis point towards currentPoint
    const float cosC = clamp(sphereRadius / distanceToSphereCenter, 0.0f, 1.0f); // Cap boundary cos theta
    const float areaCap = 2.0f * CUDART_PI_F * sphereRadius * sphereRadius * (1.0f - cosC);

    // Random sample cap: cos theta ~ U[cosC, 1], gamma ~ U[0, 2pi)
    const float u = Rand1(seed);
    const float vphi = Rand1(seed);
    const float cosT = lerp(cosC, 1.0f, u);
    const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT*cosT));
    const float phi = 2.0f * CUDART_PI_F * vphi;

    float3 t, b; BuildTangentBasis(axis, t, b);
    const float3 n = t * (cosf(phi)*sinT) + b * (sinf(phi)*sinT) + axis * cosT; // Normal
    const float3 y = sphereCenter + n * sphereRadius;

    lightSample.facetIndex_ = -1;
    lightSample.samplePoint_ = y;
    lightSample.samplePointNormal_ = n;
    
    lightSample.sampleEmissive_ = sphere.material_.texture_.baseTexture_.emissive_;
    
    lightSample.pFacet_ = 1.0f; // no facet in sphere so 1.0
    lightSample.pA_ = 1.0f / areaCap;
    lightSample.bIsValid_ = true;
    return lightSample;
}

// must PAIR with NEESampleLight
FUNCTION_MODIFIER_DEVICE_INLINE float NEEPDFOmegaFromHit(const SceneObject& lightObject, const AdditionalLightInfo& lightInfo, const float3 surfacePoint, const float3 lightPoint, const float3 lightNormal) // notice surfacePoint is actually last point because this point is light point
{
    if (lightObject.type_ == EObjectType::OBJ_CUBE)
    {
        return NEEPDFOmegaFromHit_Box(lightObject, lightInfo, surfacePoint, lightPoint, lightNormal);
    }
    else if (lightObject.type_ == EObjectType::OBJ_SPHERE)
    {
        return NEEPDFOmegaFromHit_Sphere(lightObject, lightInfo, surfacePoint, lightPoint, lightNormal);
    }
    else
    {
        return 0.0f;
    }
}

// must PAIR with NEEPDFOmegaFromHit
FUNCTION_MODIFIER_DEVICE_INLINE
LightSample NEESampleLight(
    const AdditionalLightInfo* __restrict__ sceneLightsInfo,
    const SceneObject* __restrict__ sceneObjects,
    const int lightsCount,
    const float3 samplePoint, curandState* seed)
{
    if(lightsCount <= 0)
    {
        return {};
    }
    int pickedLightIndex = lightsCount - 1;
    float pickedLightProb = 0.0f;
    {
        const float randomForLightPick = Rand1(seed);
        float acc = 0.0f;
        for(int lightIndex = 0; lightIndex < lightsCount; lightIndex++)
        {
            const float currentLightProb = sceneLightsInfo[lightIndex].selectProb_;
            acc += currentLightProb;
            if(randomForLightPick < acc)
            {
                pickedLightIndex = lightIndex;
                pickedLightProb = currentLightProb;
                break;
            }
        }
    }
    const AdditionalLightInfo& pickedLightInfo = sceneLightsInfo[pickedLightIndex];
    const SceneObject& pickedLight = sceneObjects[pickedLightInfo.objectIndex_];

    LightSample lightSample = {};
    if (pickedLight.type_ == EObjectType::OBJ_CUBE)
    {
        lightSample = NEESampleBoxAreaFacing(seed, pickedLight, samplePoint);
    }
    else if (pickedLight.type_ == EObjectType::OBJ_SPHERE)
    {
        lightSample = NEESampleSphereCapArea(seed, pickedLight, samplePoint);
    }
    if (lightSample.bIsValid_ == false)
    {
        return {};
    }
    lightSample.objectIndex_ = pickedLightInfo.objectIndex_;
    lightSample.pFacet_ = lightSample.pFacet_ * pickedLightProb;
    return lightSample;
}

FUNCTION_MODIFIER_DEVICE_INLINE float NEEProbabilityWhenGGX(const GBuffer& gbuffer, const float3 currentRayDir)
{
    const float r = clamp(gbuffer.roughness_, 0.02f, 1.0f);
    const float m = saturate_(gbuffer.metallic_);
    const float3 N = normalize(gbuffer.normal_);
    const float cosV = saturate_(dot(N, -currentRayDir));
    const float alpha = r * r;
    const float wideLobe = smooth01(0.08f * 0.08f, 0.35f * 0.35f, alpha);
    const float notGrazing = smooth01(0.05f, 0.60f, cosV);
    const float dielectricBias = 1.0f - m;
    const float specPenalty = (1.0f - wideLobe) * (1.0f - notGrazing) * (0.3f + 0.7f * m);
    const float p = 0.15f + 0.85f * ( dielectricBias * wideLobe * notGrazing * (1.0f - 0.5f * (1.0f - wideLobe))
                               + (1.0f - dielectricBias) * (wideLobe * notGrazing) * (1.0f - specPenalty) );
    return clamp(p * p, 0.05f, 0.98f);
}