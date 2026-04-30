// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"

FUNCTION_MODIFIER_INLINE float3 Gamma(float3 color) 
{
    return pow(color, 1.0f / 2.2f);
}

FUNCTION_MODIFIER_INLINE float3 unity_to_ACES(float3 x)
{
    float3x3 sRGB_2_AP0 = {
        {0.4397010f, 0.3829780f, 0.1773350f},
        {0.0897923f, 0.8134230f, 0.0967616f},
        {0.0175440f, 0.1115440f, 0.8707040f}
    };
    x = sRGB_2_AP0 * x;
    return x;
};

FUNCTION_MODIFIER_INLINE float3 ACES_to_ACEScg(float3 x)
{
    float3x3 AP0_2_AP1_MAT = {
        {1.4514393161f, -0.2365107469f, -0.2149285693f},
        {-0.0765537734f,  1.1762296998f, -0.0996759264f},
        {0.0083161484f, -0.0060324498f,  0.9977163014f}
    };
    return AP0_2_AP1_MAT * x;
};

FUNCTION_MODIFIER_INLINE float3 XYZ_2_xyY(float3 XYZ)
{
    float divisor = max(dot(XYZ, { 1,1,1 }), 1.0e-4f);
    return float3{ XYZ.x / divisor, XYZ.y / divisor, XYZ.y };
};

FUNCTION_MODIFIER_INLINE float3 xyY_2_XYZ(float3 xyY)
{
    float m = xyY.z / max(xyY.y, 1e-4f);
    float3 XYZ = float3{ xyY.x, xyY.z, (1.0f - xyY.x - xyY.y) };
    XYZ.x *= m;
    XYZ.z *= m;
    return XYZ;
};

FUNCTION_MODIFIER_INLINE float3 darkSurround_to_dimSurround(float3 linearCV)
{
    float3x3 AP1_2_XYZ_MAT = float3x3{ {0.6624541811f, 0.1340042065f, 0.1561876870f},
                               {0.2722287168f, 0.6740817658f, 0.0536895174f},
                               {-0.0055746495f, 0.0040607335f, 1.0103391003f} };
    float3 XYZ = AP1_2_XYZ_MAT * linearCV;

    float3 xyY = XYZ_2_xyY(XYZ);
    xyY.z = min(max(xyY.z, 0.0f), 65504.0f);
    xyY.z = pow(xyY.z, 0.9811f);
    XYZ = xyY_2_XYZ(xyY);

    float3x3 XYZ_2_AP1_MAT = {
        {1.6410233797f, -0.3248032942f, -0.2364246952f},
        {-0.6636628587f,  1.6153315917f,  0.0167563477f},
        {0.0117218943f, -0.0082844420f,  0.9883948585f}
    };
    return XYZ_2_AP1_MAT * XYZ;
};

FUNCTION_MODIFIER_INLINE float3 ACES(float3 color)
{

    float3x3 AP1_2_XYZ_MAT = float3x3{ {0.6624541811f, 0.1340042065f, 0.1561876870f},
                               {0.2722287168f, 0.6740817658f, 0.0536895174f},
                               {-0.0055746495f, 0.0040607335f, 1.0103391003f} };

    float3 aces = unity_to_ACES(color);

    float3 AP1_RGB2Y = float3{ 0.272229f, 0.674082f, 0.0536895f };

    float3 acescg = ACES_to_ACEScg(aces);
    float tmp = dot(acescg, AP1_RGB2Y);
    acescg = lerp(float3{ tmp,tmp,tmp }, acescg, 0.96f);
    const float a = 278.5085f;
    const float b = 10.7772f;
    const float c = 293.6045f;
    const float d = 88.7122f;
    const float e = 80.6889f;
    /*
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    */
    float3 x = acescg;
    float3 rgbPost = (x * (x * a + b)) / (x * (x * c + d) + e);
    float3 linearCV = darkSurround_to_dimSurround(rgbPost);
    tmp = dot(linearCV, AP1_RGB2Y);
    linearCV = lerp(float3{ tmp,tmp,tmp }, linearCV, 0.93f);
    float3 XYZ = AP1_2_XYZ_MAT * linearCV;
    float3x3 D60_2_D65_CAT = {
        {0.98722400f, -0.00611327f, 0.0159533f},
        {-0.00759836f,  1.00186000f, 0.0053302f},
        {0.00307257f, -0.00509595f, 1.0816800f}
    };
    XYZ = D60_2_D65_CAT * XYZ;
    float3x3 XYZ_2_REC709_MAT = {
        {3.2409699419f, -1.5373831776f, -0.4986107603f},
        {-0.9692436363f,  1.8759675015f,  0.0415550574f},
        {0.0556300797f, -0.2039769589f,  1.0569715142f}
    };
    linearCV = max({0.0f, 0.0f, 0.0f}, XYZ_2_REC709_MAT * XYZ);

    return Gamma(linearCV);
}

// XYZ2SRGBLinearD65
FUNCTION_MODIFIER_INLINE float3 XYZ2SRGBLinearD65(const float3 xyzColor)
{
    const float3x3 XYZ_2_REC709 = float3x3
    {
                {3.2406f,-1.5372f,-0.4986f},
                {-0.9689f,1.8758f,0.0415f},
                {0.0557f,-0.204f,1.057f}
    };
    const float3 rgb = XYZ_2_REC709 * xyzColor;
    return max(float3{0.0f, 0.0f, 0.0f}, rgb);
}

FUNCTION_MODIFIER_INLINE float3 SRGBLinearD65ToXYZ(const float3 rgbLinear)
{
    const float3x3 REC709_2_XYZ = float3x3
    {
            {0.4123956f, 0.3575834f, 0.1804926f},
            {0.2125862f, 0.7151703f, 0.0722005f},
            {0.0192972f, 0.1191839f, 0.9504971f}
    };

    const float3 xyz = REC709_2_XYZ * rgbLinear;
    return max(float3{0.0f, 0.0f, 0.0f}, xyz);
}