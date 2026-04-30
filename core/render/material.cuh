// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "sample.cuh"
#include "object.cuh"
#include "random.cuh"
#include "config.cuh"
#include "intersection.cuh"
#include "spectrum/spectrum_lut.cuh"

struct MediaContext
{
    bool bInsideMedia_ = false;
    float3 vertexLocation_ = {0.0f, 0.0f, 0.0f};
    int objectIndex_ = -1; 
    int depth_ = 0; // trick for infinity fully reflection
};
struct SampleContext
{
    float3 nextDirection_ = {0.0f, 0.0f, 0.0f};
    float  nextDirectionPDF_ = 1.0f;

    float3 hitEmissive_ = {0.0f, 0.0f, 0.0f};
#ifdef USE_SPECTRUM_RENDERING
    float hitBRDF_ = 0.0f; // include cosTheta
#else
    float3 hitBRDF_ = {0.0f, 0.0f, 0.0f}; // include cosTheta
#endif
    
    bool bTerminate_ = false;
    bool bHitPureLight_ = false;
    bool bDeltaSurface_ = false;
    bool bMissThinSurfaceInsideMedia_ = false;
    
    MediaContext mediaContext_;
};

struct TraceContext
{
    IntersectionContext intersectionContext_;
    SampleContext sampleContext_;
    
    EShadingModel shadingModel_;
};

// ray direction towards surface
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_DEVICE_INLINE float EvaluateBSDFWithCos(
    const float lambda,
    const SpectrumBasis& lambdaBasis,
    const float* __restrict__ lutRGB,
#else
FUNCTION_MODIFIER_DEVICE_INLINE float3 EvaluateBSDFWithCos(
#endif
    const EShadingModel shadingModel,
    const float3 rayDirection,
    const float3 lightDirection,
    const IntersectionContext& intersectionContext,
    const MediaContext& currentMediaContext)
{
    const float3 N = intersectionContext.hitNormal_;
    const float3 L = lightDirection;
    const float NoL = fmaxf(0.0f, dot(N, L));
    //GBuffer& gbuffer = intersectionContext.gbuffer_;
    GBuffer gbuffer = intersectionContext.gbuffer_;

    //混合渲染管线采样贴图
    if (gbuffer.albedoTex_ != 0)
    {
        float4 texColor = tex2D<float4>(gbuffer.albedoTex_, intersectionContext.uv_.x, intersectionContext.uv_.y);
        // sRGB 转 Linear (平方近似法，极其高效且平滑)
        gbuffer.albedo_ = make_float3(texColor.x * texColor.x, texColor.y * texColor.y, texColor.z * texColor.z);
    }

    if (shadingModel == EShadingModel::MAT_DIFFUSE)
    {
#ifdef USE_SPECTRUM_RENDERING
        return spectrum::lut::Interact(lambdaBasis, gbuffer.albedo_, lutRGB) * (NoL / CUDART_PI_F);
#else
        return gbuffer.albedo_ * (NoL / CUDART_PI_F);
#endif
    }
    else if (shadingModel == EShadingModel::MAT_LIGHT)
    {
#ifdef USE_SPECTRUM_RENDERING
        return 0.0f;
#else
        return float3{0.0f, 0.0f, 0.0f};
#endif
    }
    else if (shadingModel == EShadingModel::MAT_GGX)
    {
        const float3 V = normalize(-rayDirection);
        const float NoV = fmaxf(0.0f, dot(N, V));
        if (NoV <= 0.0f || NoL <= 0.0f)
        {
#ifdef USE_SPECTRUM_RENDERING
            return 0.0f;
#else
            return float3{0.0f, 0.0f, 0.0f};
#endif
        }
        const float alpha = gbuffer.roughness_ * gbuffer.roughness_;
        const float F0Dielectric = F0_UEStyle(gbuffer.specular_);
#ifdef USE_SPECTRUM_RENDERING
        const float albedoLambdaResponsiveness = spectrum::lut::Interact(lambdaBasis, gbuffer.albedo_, lutRGB);
        const float F0 = lerp(F0Dielectric, albedoLambdaResponsiveness, gbuffer.metallic_);
#else
        const float3 F0 = lerp(float3{F0Dielectric, F0Dielectric, F0Dielectric}, gbuffer.albedo_, gbuffer.metallic_);
#endif
        const float kD = (1.0f - gbuffer.metallic_);
#ifdef USE_SPECTRUM_RENDERING
        float SpecCos, DiffCos;
        GGXLambert_BRDF_Spectrum(V, L, N, F0, NoV, kD, alpha, albedoLambdaResponsiveness, SpecCos, DiffCos);
#else
        float3 SpecCos, DiffCos;
        GGXLambert_BRDF(V, L, N, F0, NoV, kD, alpha, gbuffer.albedo_, SpecCos, DiffCos);
#endif
        return SpecCos + DiffCos;
    }
    else if (shadingModel == EShadingModel::MAT_GLASS)
    {
#ifdef USE_SPECTRUM_RENDERING
        const float IOR = gbuffer.GetIOR(lambda);
#else
        const float IOR = gbuffer.GetIOR();
#endif
        
#ifdef USE_SPECTRUM_RENDERING
        float transmittance = 1.0f;
#else
        float3 transmittance = float3{1.0f, 1.0f, 1.0f};
#endif
        if (currentMediaContext.bInsideMedia_)
        {
            const float rayDistance = distance(currentMediaContext.vertexLocation_, intersectionContext.hitPosition_);
#ifdef USE_SPECTRUM_RENDERING
            const float albedoLambdaResponsiveness = spectrum::lut::Interact(lambdaBasis, gbuffer.albedo_, lutRGB);
            transmittance = BeerLambert_Spectrum(rayDistance, albedoLambdaResponsiveness, spectrum::SigmaAFromTiltNM(gbuffer.sigmaA_, lambda, gbuffer.tiltNumber_));
#else
            transmittance = BeerLambert(rayDistance, gbuffer.albedo_, gbuffer.sigmaA_);
#endif
        }
        const float3 nIn = (dot(N, rayDirection) > 0.0f) ? -N : N;
        const float eta = currentMediaContext.bInsideMedia_ ? IOR : 1.0f / IOR;
        const float cosI = -dot(nIn, rayDirection);
        const float sin2I = fmaxf(0.0f, 1.0f - cosI * cosI);
        const float k = 1.0f - sin2I * (eta * eta);
        if (k <= 0.0f) // TIR
        {
            return transmittance * PERFECT_DELTA_SURFACE;
        }
        const float cosT = sqrtf(k);
        const float Fr = FresnelDielectricExact(cosI, cosT, eta);
        const float3 R = reflect(rayDirection, nIn);
        const float3 T = refract(rayDirection, nIn, eta);
        const float dR = dot(L, R);
        const float dT = dot(L, T);
        if (dR >= dT)
        {
            return transmittance * Fr;
        }
        else
        {
            const float weight = (eta * eta) * (fabsf(cosT) / fmaxf(1e-6f, cosI));
            return transmittance * (1.0f - Fr) * weight;
        }
    }
#ifdef USE_SPECTRUM_RENDERING
    return 0.0f;
#else
    return float3{0.0f, 0.0f, 0.0f};
#endif
}

// ray direction towards surface
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_DEVICE_INLINE SampleContext SampleBSDF(
    const float lambda,
    const SpectrumBasis& lambdaBasis,
    const float* __restrict__ lutRGB,
#else
FUNCTION_MODIFIER_DEVICE_INLINE SampleContext SampleBSDF(
#endif
    curandState* seed,
    const EShadingModel shadingModel,
    const float3 rayDirection,
    const IntersectionContext& intersectionContext,
    const MediaContext& currentMediaContext)
{
    SampleContext result;
    const float3 N = intersectionContext.hitNormal_;
    const float3 V = -rayDirection;
    //GBuffer& gbuffer = intersectionContext.gbuffer_;
    GBuffer gbuffer = intersectionContext.gbuffer_;

    if (gbuffer.albedoTex_ != 0)
    {
        float4 texColor = tex2D<float4>(gbuffer.albedoTex_, intersectionContext.uv_.x, intersectionContext.uv_.y);
        gbuffer.albedo_ = make_float3(texColor.x * texColor.x, texColor.y * texColor.y, texColor.z * texColor.z);
    }
    if (shadingModel == EShadingModel::MAT_DIFFUSE)
    {
        const Sample sample = SampleCosineWeightedHemisphere(Rand3(seed), N);
        result.nextDirection_ = sample.direction_;
        result.nextDirectionPDF_ = sample.pdf_;
        result.hitEmissive_ = gbuffer.emissive_;
        result.mediaContext_.bInsideMedia_ = currentMediaContext.bInsideMedia_;
        result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
        return result;
    }
    else if (shadingModel == EShadingModel::MAT_LIGHT)
    {
        result.nextDirectionPDF_ = 1.0f;
        result.hitEmissive_ = gbuffer.emissive_;
        result.bTerminate_ = true;
        result.bHitPureLight_ = true;
        return result;
    }
    else if (shadingModel == EShadingModel::MAT_GLASS)
    {
#ifdef USE_SPECTRUM_RENDERING
        const float IOR = gbuffer.GetIOR(lambda);
#else
        const float IOR = gbuffer.GetIOR();
#endif
        result.bDeltaSurface_ = true;
        const bool inside = currentMediaContext.bInsideMedia_;
        const float eta = inside ? IOR : 1.0f / IOR;
        const float3 nIn = (dot(N, rayDirection) > 0.0f) ? -N : N;
        const float cosI = -dot(nIn, rayDirection);
        const float sin2I = fmaxf(0.0f, 1.0f - cosI * cosI);
        const float k = 1.0f - sin2I * (eta * eta);
        result.hitEmissive_ = float3{0,0,0};
        if (k <= 0.0f) // TIR
        {
            if (currentMediaContext.depth_ < MAX_DELTA_SURFACE_DEPTH)
            {
                result.nextDirection_ = reflect(rayDirection, nIn);
                result.nextDirectionPDF_ = 1.0f;
                result.mediaContext_.bInsideMedia_ = inside;
                result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
                return result;
            }
            else // force out
            {
                result.nextDirection_ = rayDirection;
                result.nextDirectionPDF_ = 1.0f;
                result.mediaContext_.bInsideMedia_ = !inside; // flip
                result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
                return result;
            }
        }
        const float cosT = sqrtf(k);
        const float Fr = FresnelDielectricExact(cosI, cosT, eta);
        if (Rand1(seed) < Fr)
        {
            result.nextDirection_ = reflect(rayDirection, nIn);
            result.nextDirectionPDF_ = Fr;
            result.mediaContext_.bInsideMedia_ = inside;
            result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
        }
        else
        {
            result.nextDirection_ = refract(rayDirection, nIn, eta);
            result.nextDirectionPDF_ = 1.0f - Fr;
            result.mediaContext_.bInsideMedia_ = !inside; // flip
            result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
        }
        return result;
    }
    else if (shadingModel == EShadingModel::MAT_GGX)
    {
        result.hitEmissive_ = gbuffer.emissive_;
        const float NoV = fmaxf(0.0f, dot(N, V));
        if (NoV <= 0.0f)
        {
            result.bTerminate_ = true;
            return result;
        }
        const float alpha = gbuffer.roughness_ * gbuffer.roughness_;
        const float F0Dielectric = F0_UEStyle(gbuffer.specular_);
#ifdef USE_SPECTRUM_RENDERING
        const float albedoLambdaResponsiveness = spectrum::lut::Interact(lambdaBasis, gbuffer.albedo_, lutRGB);
        const float F0 = lerp(F0Dielectric, albedoLambdaResponsiveness, gbuffer.metallic_);
        const float pSpec = clamp(F0, 0.1f, 0.9f);
#else
        const float3 F0 = lerp(float3{F0Dielectric, F0Dielectric, F0Dielectric}, gbuffer.albedo_, gbuffer.metallic_);
        const float pSpec = clamp(luminance(F0), 0.1f, 0.9f);
#endif
        //
        const float u = Rand1(seed);
        float3 L;
        float pdfSpec = 0.0f;
        float pdfDiff = 0.0f;
        if (u < pSpec)
        {
            const Sample sample = SampleGGXBoundedVNDF(V, N, alpha, Rand3(seed));
            L = sample.direction_;
            const float NoL = fmaxf(0.0f, dot(N, L));
            if (NoL <= 0.0f)
            {
                result.bTerminate_ = true;
                return result;
            }
            pdfSpec = sample.pdf_;
            pdfDiff = NoL / CUDART_PI_F;

            const float pdfMix = pSpec * pdfSpec + (1.0f - pSpec) * pdfDiff;
            result.nextDirection_ = L;
            result.nextDirectionPDF_ = pdfMix;
        }
        else
        {
            const Sample sample = SampleCosineWeightedHemisphere(Rand3(seed), N);
            L = sample.direction_;
            pdfDiff = sample.pdf_;
            pdfSpec = GGXBoundedVNDF_PDF(V, N, L, alpha);

            const float pdfMix = pSpec * pdfSpec + (1.0f - pSpec) * pdfDiff;
            result.nextDirection_ = L;
            result.nextDirectionPDF_ = pdfMix;
        }
        result.mediaContext_.bInsideMedia_ = currentMediaContext.bInsideMedia_;
        result.mediaContext_.vertexLocation_ = intersectionContext.hitPosition_;
        return result;
    }
    return result;
}

// light direction towards light
// ray direction towards surface
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_INLINE float BSDF_PDF(
    const SpectrumBasis& lambdaBasis,
    const float* __restrict__ lutRGB,
#else
FUNCTION_MODIFIER_INLINE float BSDF_PDF(
#endif
    const EShadingModel shadingModel,
    const GBuffer& gbuffer,
    const float3 surfacePoint,
    const float3 surfaceNormal,
    const float3 rayDirection,
    const float3 lightDirection)
{
    const float3 N = surfaceNormal;
    const float3 V = -rayDirection;
    const float3 L = lightDirection;
    const float NoL = fmaxf(0.0f, dot(N, L));
    if (shadingModel == EShadingModel::MAT_DIFFUSE)
    {
        return NoL > 0.0f ? (NoL * (1.0f / CUDART_PI_F)) : 0.0f;
    }
    else if (shadingModel == EShadingModel::MAT_LIGHT)
    {
        return 0.0f;
    }
    else if (shadingModel == EShadingModel::MAT_GLASS)
    {
        return 0.0f;
    }
    else if (shadingModel == EShadingModel::MAT_GGX)
    {
        const float NoV = fmaxf(0.0f, dot(N, V));
        if (NoV <= 0.0f || NoL <= 0.0f)
        {
            return 0.0f;
        }
        const float alpha = gbuffer.roughness_ * gbuffer.roughness_;
        const float F0Dielectric = F0_UEStyle(gbuffer.specular_);
#ifdef USE_SPECTRUM_RENDERING
        const float albedoLambdaResponsiveness = spectrum::lut::Interact(lambdaBasis, gbuffer.albedo_, lutRGB);
        const float F0 = lerp(F0Dielectric, albedoLambdaResponsiveness, gbuffer.metallic_);
        const float pSpec = clamp(F0, 0.1f, 0.9f);
#else
        const float3 F0 = lerp(float3{F0Dielectric, F0Dielectric, F0Dielectric}, gbuffer.albedo_, gbuffer.metallic_);
        const float pSpec = clamp(luminance(F0), 0.1f, 0.9f);
#endif
        const float pdfSpec = GGXBoundedVNDF_PDF(V, N, L, alpha);
        const float pdfDiff = NoL * (1.0f / CUDART_PI_F);
        const float pdfMix = pSpec * pdfSpec + (1.0f - pSpec) * pdfDiff;
        return fmaxf(0.0f, pdfMix);
    }
    return 0.0f;
}

// ray direction towards surface
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_DEVICE_INLINE SampleContext EvaluateMaterial(
    const float lambda,
    const SpectrumBasis& lambdaBasis,
    const float* __restrict__ lutRGB,
#else
FUNCTION_MODIFIER_DEVICE_INLINE SampleContext EvaluateMaterial(
#endif
    curandState* seed,
    const EShadingModel shadingModel,
    float3 rayDirection,
    const IntersectionContext& intersectionContext,
    const MediaContext& currentMediaContext)
{
#ifdef USE_SPECTRUM_RENDERING
    SampleContext sampleResult = SampleBSDF(lambda, lambdaBasis, lutRGB, seed, shadingModel, rayDirection, intersectionContext, currentMediaContext);
#else
    SampleContext sampleResult = SampleBSDF(seed, shadingModel, rayDirection, intersectionContext, currentMediaContext);
#endif
    sampleResult.mediaContext_.objectIndex_ = intersectionContext.objectIndex_;
    sampleResult.mediaContext_.depth_ = currentMediaContext.depth_ + 1;

#ifdef USE_SPECTRUM_RENDERING
    sampleResult.hitBRDF_ = EvaluateBSDFWithCos(
        lambda, lambdaBasis, lutRGB,
#else
    sampleResult.hitBRDF_ = EvaluateBSDFWithCos(
#endif
        shadingModel,
        rayDirection,
        sampleResult.nextDirection_,
        intersectionContext,
        currentMediaContext
    );
    return sampleResult;
}