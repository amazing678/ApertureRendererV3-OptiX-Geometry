// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "brdf.cuh"
#include <cuda_runtime.h>

struct Sample
{
    float3 direction_;
    float pdf_;
    FUNCTION_MODIFIER_INLINE Sample() : direction_{0,0,0}, pdf_(0) {}
    FUNCTION_MODIFIER_INLINE Sample(const float3 direction, const float pdf) : direction_(direction), pdf_(pdf) {}
};

FUNCTION_MODIFIER_INLINE float2 Roberts2(const int n)
{
    const float g = 1.32471795724474602596f;
    const float2 a = float2{ 1.0f / g, 1.0f / (g * g) };
    return frac(a * static_cast<float>(n) + 0.5f);
}
FUNCTION_MODIFIER_INLINE float3 Roberts3(const int n)
{
    const float g = 1.32471795724474602596f;
    const float3 a = float3{ 1.0f / g, 1.0f / (g * g) , 1.0f / (g * g * g) };
    return frac(a * static_cast<float>(n) + 0.5f);
}
FUNCTION_MODIFIER_INLINE Sample SampleUniformHemisphere(const float2 e)
{
    const float phi = 2.0f * CUDART_PI_F * e.x;
    const float z = e.y;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));

    float s, c; 
    sincosf(phi, &s, &c);

    const float3 dir = float3{ r * c, r * s, z };
    const float pdf = 1.0f / (2.0f * CUDART_PI_F);
    return {dir, pdf};
}

FUNCTION_MODIFIER_INLINE Sample SampleUniformHemisphereOriented(const float3 e, const float3 normal)
{
    const Sample sLocal = SampleUniformHemisphere(make_float2(e.x, e.y)); // pdf = 1/(2π)
    float3 T, B;
    BuildTangentBasisRandom(normalize(normal), T, B, e.z);
    const float3 dirW = normalize(sLocal.direction_.x * T +
                                  sLocal.direction_.y * B +
                                  sLocal.direction_.z * normalize(normal));
    return { dirW, sLocal.pdf_ };
}

FUNCTION_MODIFIER_INLINE Sample SampleUniformSphere(const float2 e)
{
    const float phi = 2.0f * CUDART_PI_F * e.x;
    const float cosTheta = 1.0f - 2.0f * e.y;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float3 dir = float3{sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta};
    const float pdf = 1.0f / (4.0f * CUDART_PI_F);
    return {dir, pdf};
}

FUNCTION_MODIFIER_INLINE float2 ConcentricSampleDisk(const float2 u)
{
    const float sx = 2.0f * u.x - 1.0f;
    const float sy = 2.0f * u.y - 1.0f;
    if (sx == 0.0f && sy == 0.0f)
    {
        return float2{0.0f, 0.0f};
    }
    float r;
    float phi;
    if (fabsf(sx) > fabsf(sy))
    {
        r = sx;
        phi = (CUDART_PI_F/4.0f) * (sy / fmaxf(fabsf(sx), 1e-20f));
    }
    else
    {
        r = sy;
        phi = (CUDART_PI_F/2.0f) - (CUDART_PI_F/4.0f) * (sx / fmaxf(fabsf(sy), 1e-20f));
    }
    const float c = cosf(phi);
    const float s = sinf(phi);
    return float2{ r * c, r * s };
}

FUNCTION_MODIFIER_INLINE Sample SampleCosineWeightedHemisphere(const float3 e, const float3 normal)
{
    /*
    const Sample sampleSphere = SampleUniformSphere(e);
    const float3 dir = normalize(sampleSphere.direction_ + normal);
    const float cosTheta = fmaxf(0.0f, dot(normal, dir));
    const float pdf = cosTheta / CUDART_PI_F;
    return {dir, pdf};
    */
    float3 T, B;
    //BuildTangentBasis(normal, T, B);
    BuildTangentBasisRandom(normal, T, B, e.z);
    const float2 d = ConcentricSampleDisk({e.x, e.y});
    const float x = d.x;
    const float y = d.y;
    const float z = sqrtf(fmaxf(0.0f, 1.0f - x*x - y*y));

    const float3 dir = x * T + y * B + z * normal;
    const float pdf  = z / CUDART_PI_F;
    return { normalize(dir), pdf };
}

FUNCTION_MODIFIER_INLINE float HenyeyGreensteinPDF(const float cosTheta, const float g)
{
    const float g2  = g * g;
    //const float den = powf(1.0f + g2 - 2.0f * g * cosTheta, 1.5f);
    float x = 1.0f + g2 - 2.0f * g * cosTheta;
    x = fmaxf(x, 1e-20f);
    float den = x * sqrtf(x);
    return (1.0f - g2) / (4.0f * CUDART_PI_F * den);
}
FUNCTION_MODIFIER_INLINE float SampleHenyeyGreensteinBase(const float e, const float g)
{
    if (fabsf(g) < 0.0001f)
    {
        return e * 2.0f - 1.0f;
    }
    const float g2 = g * g;
    const float t0 = (1.0f - g2) / (1.0f - g + 2.0f * g * e);
    const float cosAng = (1.0f + g2 - t0 * t0) / (2.0f * g);
    return cosAng;
}

FUNCTION_MODIFIER_INLINE Sample SampleHenyeyGreenstein(const float2 e, const float3 v, const float g)
{
    const float cosTheta = SampleHenyeyGreensteinBase(e.x, g);
    const float phi = 2 * CUDART_PI_F * e.y;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    float3 w  = normalize(v);
    float3 a = (fabsf(w.z) > SQRT_05) ? float3{0,1,0} : float3{0,0,1};
    float3 v2 = normalize(cross(w, a));
    float3 v3 = cross(v2, w);

    const float3 dir = v3 * (sinTheta * cosf(phi)) + v2 * (sinTheta * sinf(phi)) + w * cosTheta;
    const float pdf = HenyeyGreensteinPDF(cosTheta, g);
    return {dir, pdf};
}


//https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
FUNCTION_MODIFIER_INLINE Sample SampleGGXAnisotropicBoundedVNDF(const float3 V, const float3 N, const float3 X, const float3 Y, const float2 alpha, const float2 rand)
{
    const float NoV = dot(N, V); // z
    const float XoV = dot(X, V); // x
    const float YoV = dot(Y, V); // y
    const float3 localV = {XoV, YoV, NoV};
    const float3 localVStd = normalize(float3{ XoV * alpha.x, YoV * alpha.y, NoV}) ;
    const float phi = 2.0f * CUDART_PI_F * rand.x;
    const float a = saturate_(fminf( alpha.x, alpha.y));
    const float s = 1.0f + length(float2{XoV, YoV});
    const float a2 = a * a;
    const float s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * NoV * NoV);
    const float b = NoV > 0 ? k * localVStd.z : localVStd.z ;
    const float z = fmaf(1.0f - rand .y , 1.0f + b , -b); // mad
    const float sinTheta = sqrtf(saturate_(1.0f - z * z));
    const float3 oStd = {sinTheta * cos(phi), sinTheta * sin(phi), z};
    const float3 mStd = localVStd + oStd ;
    const float3 localH = normalize(float3{mStd.x * alpha.x, mStd.y * alpha.y, mStd.z});
    const float3 localL = reflect(-localV, localH);
    const float3 dir = localL.x * X + localL.y * Y + localL.z * N;

    const float ndf = GGX_D_Anisotropic(alpha, localH.z, localH.x, localH.y);
    const float2 ai = {XoV * alpha.x, YoV * alpha.y};
    const float len2 = dot(ai , ai);
    const float t = sqrt(len2 + NoV * NoV) ;
    const float pdf = ndf / (2.0f * (k * NoV + t));
    return {dir, pdf};
}

FUNCTION_MODIFIER_INLINE Sample SampleGGXBoundedVNDF(const float3 V, const float3 N, const float alpha, const float3 rand)
{
    float3 X;
    float3 Y;
    //BuildTangentBasis(N, X, Y);
    BuildTangentBasisRandom(N, X, Y, rand.z);
    const float XoV = dot(X, V); // x
    const float YoV = dot(Y, V); // y
    const float NoV = dot(N, V); // z
    const float3 localV = {XoV, YoV, NoV};
    const float NoV2 = fminf(1.0f, NoV * NoV); // z*z
    const float a2 = alpha * alpha;
    
    // normalize(float3{ XoV * alpha.x, YoV * alpha.y, NoV})
    // = float3{ XoV * alpha.x, YoV * alpha.y, NoV} / (XoV * alpha.x * XoV * alpha.x + YoV * alpha.y * YoV * alpha.y + NoV * NoV)
    // = float3{ XoV * alpha.x, YoV * alpha.y, NoV} / ((XoV * XoV + YoV * YoV) * a2 + NoV2)
    // = float3{ XoV * alpha.x, YoV * alpha.y, NoV} / ((1.0 - NoV2) * a2 + NoV2)
    // = float3{ XoV * alpha.x, YoV * alpha.y, NoV} / (a2 - NoV2 * a2 + NoV2)
    // = float3{ XoV * alpha.x, YoV * alpha.y, NoV} / (a2 + (1 - a2) * NoV2)
    const float3 localRayDirectionStd = float3{XoV * alpha, YoV * alpha, NoV} / sqrtf(a2 + (1.0f - a2) * NoV2);
    const float phi = 2.0f * CUDART_PI_F * rand.x;
    
    // 1.0f + length(float2{XoV, YoV}) = 1.0 + sqrt(1.0 - NoV2)
    const float s = 1.0f + sqrtf(1.0f - NoV2);
    const float s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * NoV2);
    const float b = NoV > 0 ? k * localRayDirectionStd.z : localRayDirectionStd.z ;
    const float z = fmaf(1.0f - rand .y , 1.0f + b , -b); // mad
    const float sinTheta = sqrtf(saturate_(1.0f - z * z));
    
    const float3 oStd = {sinTheta * cos(phi), sinTheta * sin(phi), z};
    const float3 mStd = localRayDirectionStd + oStd ;
    const float3 localH = normalize(float3{mStd.x * alpha, mStd.y * alpha, mStd.z});
    const float3 localL = reflect(-localV, localH); // v - 2.0f * dot(v, n) * n // 2.0f * dot({XoV,YoV,NoV}, localH) * localH - float3{XoV,YoV,NoV}
    const float3 dir = localL.x * X + localL.y * Y + localL.z * N;

    const float ndf = GGX_D(alpha, localH.z);
    // ai = {XoV * alpha, YoV * alpha}
    // len2 = dot(ai , ai) = XoV * alpha * XoV * alpha + YoV * alpha * YoV * alpha = (1.0 - NoV2) * a2
    const float len2 = (1.0f - NoV2) * a2;
    const float t = sqrtf(len2 + NoV2) ;
    const float pdf = ndf / (2.0f * (k * NoV + t));
    return {dir, pdf};
}

FUNCTION_MODIFIER_INLINE float GGXBoundedVNDF_PDF(const float3 V, const float3 N, const float3 L, const float alpha)
{
    const float NoV = dot(N, V);
    const float NoL = dot(N, L);
    if (NoV <= 0.0f || NoL <= 0.0f)
    {
        return 0.0f;
    }
    const float3 H = normalize(V + L);
    const float NoV2 = fminf(1.0f, NoV * NoV); // z*z
    const float a2 = alpha*alpha;
    
    const float s = 1.0f + sqrtf(1.0f - NoV2);
    const float s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * NoV2);
    
    const float NoH = fmaxf(0.0f, dot(N, H));
    const float ndf = GGX_D(alpha, NoH);
    
    const float len2 = (1.0f - NoV2) * a2;
    const float t = sqrtf(len2 + NoV2);
    const float pdf = ndf / (2.0f * (k * NoV + t));

    return fmaxf(pdf, 1e-12f);
}

// normal ggx ndf H, for glass
FUNCTION_MODIFIER_INLINE float3 SampleGGX_NDF_H(const float alpha, const float2 u)
{
    // Trowbridge-Reitz (GGX)
    // tan^2(theta) = a^2 * u / (1-u)  ->  cos theta = 1 / sqrt(1 + tan^2(theta))
    const float a = fmaxf(alpha, 1e-6f);
    const float phi = 2.0f * CUDART_PI_F * u.y;
    const float u1 = fmaxf(u.x, 1e-8f);
    const float tan2 = (a * a) * (u1) / (1.0f - u1);
    const float cosTheta = rsqrtf(1.0f + tan2);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    return float3{cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
}