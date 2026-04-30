// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "render/object.cuh"
#include "vector.cuh"
#include "render/material.cuh"


namespace denoise
{
    __align__(16) struct ScreenStatisticsBuffer
    {
#ifdef __CUDA_ARCH__
        uint32_t num_;
        float3 mean_;
        float3 M2DivNum_;
        float3 M3DivNum_;
#else
        uint32_t num_ = 0u;
        float3 mean_ = {0.0f,0.0f,0.0f};
        float3 M2DivNum_ = {0.0f,0.0f,0.0f};
        float3 M3DivNum_ = {0.0f,0.0f,0.0f};
#endif
    };
    //
    FUNCTION_MODIFIER_INLINE float3 BoxCox05(const float3 x)
    {
        return (pow(x, 0.5f) - 1.0f) * 2.0f;
    }
    
    FUNCTION_MODIFIER_INLINE float3 BoxCox(const float3 x)
    {
        return BoxCox05(max(x, 0.0f));
    }
    
    FUNCTION_MODIFIER_INLINE void RecordScreenStatisticBuffer(const float3 xyz, denoise::ScreenStatisticsBuffer* __restrict__ hist = nullptr)
    {
        const uint32_t n0 = hist->num_;
        const uint32_t n1 = n0 + 1u;
        hist->num_ = n1;
        
        const float3 x_bc = BoxCox(xyz);

        const float3 mean0 = hist->mean_;
        const float3 delta = x_bc - mean0;
        const float invN1 = 1.0f / static_cast<float>(n1);
        const float3 deltaDivN = delta * invN1;
        
        const float3 mean1 = mean0 + deltaDivN;
        
        const double M2_0_x = static_cast<double>(hist->M2DivNum_.x) * n0;
        const double M2_0_y = static_cast<double>(hist->M2DivNum_.y) * n0;
        const double M2_0_z = static_cast<double>(hist->M2DivNum_.z) * n0;
        const double M2_1_x = M2_0_x + delta.x * (delta.x - deltaDivN.x);
        const double M2_1_y = M2_0_y + delta.y * (delta.y - deltaDivN.y);
        const double M2_1_z = M2_0_z + delta.z * (delta.z - deltaDivN.z);
        
        const double M3_0_x = static_cast<double>(hist->M3DivNum_.x) * n0;
        const double M3_0_y = static_cast<double>(hist->M3DivNum_.y) * n0;
        const double M3_0_z = static_cast<double>(hist->M3DivNum_.z) * n0;
        const float3 delta2 = delta * delta; // delta^2
        const float3 deltaDivNPow2 = deltaDivN * deltaDivN; // (delta/n)^2
        const double M3_1_x = M3_0_x - 3.0 * (deltaDivN.x * M2_1_x) + delta.x * (delta2.x - deltaDivNPow2.x);
        const double M3_1_y = M3_0_y - 3.0 * (deltaDivN.y * M2_1_y) + delta.y * (delta2.y - deltaDivNPow2.y);
        const double M3_1_z = M3_0_z - 3.0 * (deltaDivN.z * M2_1_z) + delta.z * (delta2.z - deltaDivNPow2.z);
        
        hist->mean_ = mean1;
        hist->M2DivNum_ = float3{static_cast<float>(M2_1_x / n1), static_cast<float>(M2_1_y / n1), static_cast<float>(M2_1_z / n1)};
        hist->M3DivNum_ = float3{static_cast<float>(M3_1_x / n1), static_cast<float>(M3_1_y / n1), static_cast<float>(M3_1_z / n1)};
    }
}