// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include <cuda_runtime.h>

#include "color.cuh"
#include "spectrum/spectrum.cuh"
#include "spectrum/spectrum_basis.cuh"

#define SQRT_05 0.70710678f


// Beer Lambert
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 BeerLambert(float distance, float3 albedo, float sigmaA)
{
    return exp(distance * sigmaA * log(min(max(albedo, 1.0e-12f), 1.0f)));
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float BeerLambert_Spectrum(float distance, float albedoLambdaResponsiveness, float sigmaA)
{
    return expf(distance * sigmaA * logf(fminf(fmaxf(albedoLambdaResponsiveness, 1.0e-12f), 1.0f)));
}

// Fresnel for glass
[[nodiscard]] FUNCTION_MODIFIER_INLINE float FresnelDielectricExact(float cosI, float cosT, float eta/* n_in / n_out */)
{
    const float invE = 1.0f / eta;
    const float rPar  = (invE * cosI - cosT) / (invE * cosI + cosT);
    const float rPerp = (cosI - invE * cosT) / (cosI + invE * cosT);
    return 0.5f * (rPar * rPar + rPerp * rPerp);
}

// GGX
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 FresnelSchlick(const float3 F0, const float cosX)
{
    const float c = clamp(1.0f - cosX, 0.0f, 1.0f);
    const float c5 = c * c * c * c * c;
    return F0 + (float3{1.0f,1.0f,1.0f} - F0) * c5;
}
[[nodiscard]] FUNCTION_MODIFIER_INLINE float FresnelSchlick(const float F0, const float cosX)
{
    const float c = clamp(1.0f - cosX, 0.0f, 1.0f);
    const float c5 = c * c * c * c * c;
    return F0 + (1.0f - F0) * c5;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float GGX_G1_Smith(const float alpha, const float NoX)
{
    const float a = alpha;
    const float denom = NoX + sqrtf(a*a + (1.0f - a*a) * NoX * NoX);
    return (2.0f * NoX) / fmaxf(denom, 1e-7f);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float GGX_G(const float alpha, const float NoL, const float NoV)
{
    return GGX_G1_Smith(alpha, NoL) * GGX_G1_Smith(alpha, NoV);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float GGX_G_Correlated(const float alpha, const float NoL, const float NoV)
{
    const float a2 = alpha * alpha;
    const float gv = NoL * sqrtf(a2 + (1.0f - a2) * NoV * NoV);
    const float gl = NoV * sqrtf(a2 + (1.0f - a2) * NoL * NoL);
    const float denom = gv + gl;
    return (denom > 0.0f) ? (2.0f * NoL * NoV) / denom : 0.0f;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float DielectricF0FromIOR(float IOR)
{
    const float r = (IOR - 1.0f) / (IOR + 1.0f);
    return r * r;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float F0_UEStyle(const float specular01)
{
    const float s = clamp(specular01, 0.0f, 1.0f);
    const float f = 0.08f * s;
    return f;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float GGX_D(const float alpha, const float NoH)
{
    const float a2 = alpha * alpha;
    const float d  = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(CUDART_PI_F * d * d, 1e-12f);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float GGX_D_Anisotropic(const float2 alpha, const float NoH, const float XoH, const float YoH)
{
    const float2 a2 = alpha * alpha;
    const float d = XoH * XoH / a2.x + YoH * YoH / a2.y + NoH * NoH;
    return 1.0f / (CUDART_PI_F * alpha.x * alpha.y * d * d);
}

FUNCTION_MODIFIER_INLINE void GGXLambert_BRDF(const float3 V, const float3 L, const float3 N,
    const float3 F0,
    const float NoV, const float kD, const float alpha,
    const float3 albedo,
    float3& outSpecCos, float3& outDiffCos)
{
    const float NoL = fmaxf(0.0f, dot(N, L));
    const float3 Hn = normalize(V + L);
    const float NoH = fmaxf(0.0f, dot(N, Hn));
    const float LoH = fmaxf(0.0f, dot(L, Hn));
    const float3 F = FresnelSchlick(F0, LoH);
    const float D = GGX_D(alpha, NoH);
    //const float  G = GGX_G(alpha, NoL, NoV);
    const float G = GGX_G_Correlated(alpha, NoL, NoV);
    outSpecCos = (NoL > 0.0f) ? (F * D * G) / fmaxf(4.0f * NoV, 1e-7f) : float3{0.0f,0.0f,0.0f};
    outDiffCos = kD * (float3{1.0f,1.0f,1.0f} - F) * albedo * (NoL / CUDART_PI_F);
}

FUNCTION_MODIFIER_INLINE void GGXLambert_BRDF_Spectrum(const float3 V, const float3 L, const float3 N,
    const float F0,
    const float NoV, const float kD, const float alpha,
    const float albedoLambdaResponsiveness,
    float& outSpecCos, float& outDiffCos)
{
    const float NoL = fmaxf(0.0f, dot(N, L));
    const float3 Hn = normalize(V + L);
    const float NoH = fmaxf(0.0f, dot(N, Hn));
    const float LoH = fmaxf(0.0f, dot(L, Hn));
    const float F = FresnelSchlick(F0, LoH);
    const float D = GGX_D(alpha, NoH);
    //const float  G = GGX_G(alpha, NoL, NoV);
    const float G = GGX_G_Correlated(alpha, NoL, NoV);
    outSpecCos = (NoL > 0.0f) ? (F * D * G) / fmaxf(4.0f * NoV, 1e-7f) : 0.0f;
    outDiffCos = kD * (1.0f - F) * albedoLambdaResponsiveness * (NoL / CUDART_PI_F);
}