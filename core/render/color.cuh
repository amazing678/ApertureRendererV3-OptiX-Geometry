// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include <cuda_runtime.h>
#include "spectrum/spectrum_basis.cuh"


struct SpectrumBasis
{
    float __align__(16) spectrumBasis_[spectrum::query::KERNEL]{};
    FUNCTION_MODIFIER_INLINE float query(const SpectrumBasis& lambdaBasis) const
    {
        return spectrum::query::SpectralInteract(spectrumBasis_, lambdaBasis.spectrumBasis_);
    }
};

struct Emission
{
    SpectrumBasis emissiveSpectrum_{};
    float emissiveScale_ = 0.0f;
    FUNCTION_MODIFIER_INLINE float query(const SpectrumBasis& lambdaBasis) const
    {
        return emissiveScale_ * spectrum::query::SpectralInteract(emissiveSpectrum_.spectrumBasis_, lambdaBasis.spectrumBasis_);
    }
};

struct color
{
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Red() {return {1.0f, 0.0f, 0.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Green() {return {0.0f, 1.0f, 0.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Blue() {return {0.0f, 0.0f, 1.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Yellow() {return {1.0f, 1.0f, 0.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Purple() {return {1.0f, 0.0f, 1.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Cyan() {return {0.0f, 1.0f, 1.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 White() {return {1.0f, 1.0f, 1.0f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Pink() {return {1.0f, 0.25f, 0.5f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Orange() {return {1.0f, 0.5f, 0.25f};}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 Azure() {return float3{ 0.25,0.5,1.0 };}
    
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 RedLighten(float lighten = 0.0f) {return lerp(Red(), 1.0f, lighten);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 GreenLighten(float lighten = 0.0f) {return lerp(Green(), 1.0f, lighten);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 BlueLighten(float lighten = 0.0f) {return lerp(Blue(), 1.0f, lighten);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 YellowLighten(float lighten = 0.0f) {return lerp(Yellow(), 1.0f, lighten);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 PurpleLighten(float lighten = 0.0f) {return lerp(Purple(), 1.0f, lighten);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 CyanLighten(float lighten = 0.0f) {return lerp(Cyan(), 1.0f, lighten);}
    
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 RedDarken(float darken = 0.0f) {return lerp(Red(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 GreenDarken(float darken = 0.0f) {return lerp(Green(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 BlueDarken(float darken = 0.0f) {return lerp(Blue(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 YellowDarken(float darken = 0.0f) {return lerp(Yellow(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 PurpleDarken(float darken = 0.0f) {return lerp(Purple(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 CyanDarken(float darken = 0.0f) {return lerp(Cyan(), 0.0f, darken);}
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 WhiteDarken(float darken = 0.0f) {return lerp(White(), 0.0f, darken);}

    
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 HSV2RGB(float h, float s, float v)
    {
        if (s <= 1e-6f)
        {
            return make_float3(v,v,v);
        }
        h = fmodf(fmaxf(h,0.0f), 1.0f) * 6.0f;
        const int i = static_cast<int>(floorf(h));
        const float f = h - static_cast<float>(i);
        const float p = v * (1.0f - s);
        const float q = v * (1.0f - s * f);
        const float t = v * (1.0f - s * (1.0f - f));
        if (i==0)
        {
            return make_float3(v,t,p);
        }
        else if (i==1)
        {
            return make_float3(q,v,p);
        }
        else if (i==2)
        {
            return make_float3(p,v,t);
        }
        else if (i==3)
        {
            return make_float3(p,q,v);
        }
        else if (i==4)
        {
            return make_float3(t,p,v);
        }
        else
        {
            return make_float3(v,p,q);
        }
    }

    enum class EHeatMap
    {
        VIRIDIS = 0,
    };
    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 HeatMapViridis(float t)
    {
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        const float3 c0 = make_float3(0.267004f, 0.004874f, 0.329415f); // t=0.00
        const float3 c1 = make_float3(0.229739f, 0.322361f, 0.545706f); // t=0.25
        const float3 c2 = make_float3(0.127568f, 0.566949f, 0.550556f); // t=0.50
        const float3 c3 = make_float3(0.369214f, 0.788888f, 0.382914f); // t=0.75
        const float3 c4 = make_float3(0.993248f, 0.906157f, 0.143936f); // t=1.00

        float a = t * 4.0f;
        int i = static_cast<int>(floorf(a));
        float f = a - static_cast<float>(i);
        if (i <= 0)
        {
            return make_float3(
                c0.x + (c1.x - c0.x) * f,
                c0.y + (c1.y - c0.y) * f,
                c0.z + (c1.z - c0.z) * f);
        }
        else if (i == 1)
        {
            return make_float3(
                c1.x + (c2.x - c1.x) * f,
                c1.y + (c2.y - c1.y) * f,
                c1.z + (c2.z - c1.z) * f);
        }
        else if (i == 2)
        {
            return make_float3(
                c2.x + (c3.x - c2.x) * f,
                c2.y + (c3.y - c2.y) * f,
                c2.z + (c3.z - c2.z) * f);
        }
        else if (i >= 3)
        {
            return make_float3(
                c3.x + (c4.x - c3.x) * f,
                c3.y + (c4.y - c3.y) * f,
                c3.z + (c4.z - c3.z) * f);
        }
        return c4;
    }

    [[nodiscard]] FUNCTION_MODIFIER_STATIC float3 ProbeValueToRGB(const float value, const EHeatMap heatMap)
    {
        if(heatMap == EHeatMap::VIRIDIS)
        {
            return HeatMapViridis(value);
        }
        return HeatMapViridis(value);
    }
};