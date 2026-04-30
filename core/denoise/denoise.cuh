// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "denoise_statistic.cuh"
#include "render/object.cuh"
#include "vector.cuh"
#include "render/material.cuh"

#include "t_quantile.cuh"

namespace denoise
{
    struct FilterParams
    {
        static constexpr float sigmaAlbedo_ = 1.0f; // smaller the sharper
        static constexpr float sigmaNormal_ = 0.5f;
        static constexpr float sigmaDepth_ = 0.25f;
        static constexpr float sigmaMetallic_ = 0.5f;
        static constexpr float sigmaRough_ = 0.5f;
        static constexpr float sigmaShadingModel_ = 0.1f;
        static constexpr float SigmaSpatial_ = 2.0f;
        
        static constexpr float inv2Sigma2Albedo_ = 1.0f / (2.0f * sigmaAlbedo_ * sigmaAlbedo_);
        static constexpr float inv2Sigma2Normal_ = 1.0f / (2.0f * sigmaNormal_ * sigmaNormal_);
        static constexpr float inv2Sigma2Depth_ = 1.0f / (2.0f * sigmaDepth_ * sigmaDepth_);
        static constexpr float inv2Sigma2Metallic_ = 1.0f / (2.0f * sigmaMetallic_ * sigmaMetallic_);
        static constexpr float inv2Sigma2Rough_ = 1.0f / (2.0f * sigmaRough_ * sigmaRough_);
        static constexpr float inv2Sigma2ShadingModel_ = 1.0f / (2.0f * sigmaShadingModel_ * sigmaShadingModel_);
        static constexpr float inv2Sigma2Spatial_ = 1.0f / (2.0f * SigmaSpatial_ * SigmaSpatial_);
    };
    
    constexpr static uint64_t MAX_SHADING_MODEL = static_cast<uint64_t>(EShadingModel::MAT_MAX);
    __align__(16) struct ScreenGBuffer
    {
#ifdef __CUDA_ARCH__
        half albedoX_;
        half albedoY_;
        half albedoZ_;
#else
        half albedoX_ = 0.0f;
        half albedoY_ = 0.0f;
        half albedoZ_ = 0.0f;
#endif
        
#ifdef __CUDA_ARCH__
        half normalX_;
        half normalY_;
        half normalZ_; 
#else
        half normalX_ = 0.0f;
        half normalY_ = 0.0f;
        half normalZ_ = 1.0f; 
#endif
        // 3 float
#ifdef __CUDA_ARCH__
        half metallic_;
        half roughness_;
#else
        half metallic_ = 0.0f;
        half roughness_ = 0.0f;
#endif
        // 4 float
#ifdef __CUDA_ARCH__
        float depth_;
#else
        float depth_ = 0.0f;
#endif
        // 5 float
#ifdef __CUDA_ARCH__
        uint64_t validFrameIndex_; 
        uint64_t frameIndex_;
#else
        uint64_t validFrameIndex_ = 0; 
        uint64_t frameIndex_ = 0;
#endif
        // 9 float
#ifdef __CUDA_ARCH__
        __align__(16) half shadingModel_[MAX_SHADING_MODEL + 1ull]; // include invalid
#else
        __align__(16) half shadingModel_[MAX_SHADING_MODEL + 1ull] = {}; // include invalid
#endif
    };
#ifdef __CUDA_ARCH__
static_assert(std::is_trivially_copyable<ScreenGBuffer>::value, "must be POD");
#endif
    
    FUNCTION_MODIFIER_INLINE void RecordShadingModel(const EShadingModel currentShadingModel, ScreenGBuffer* __restrict__ histogram)
    {
        if(histogram->frameIndex_ == 0)
        {
            CUDA_UNROLL
            for(int i=0; i<MAX_SHADING_MODEL+1; i++)
            {
                histogram->shadingModel_[i] = 0.0f;
            }
        }
        const float affixAlpha = 1.0f / static_cast<float>(histogram->frameIndex_ + 1ull);
        const float preAlpha = static_cast<float>(histogram->frameIndex_) / static_cast<float>(histogram->frameIndex_ + 1ull);
        CUDA_UNROLL
        for(int i=0; i<MAX_SHADING_MODEL+1; i++)
        {
            histogram->shadingModel_[i] =
                static_cast<half>(static_cast<float>(histogram->shadingModel_[i])* preAlpha);
        }
        histogram->shadingModel_[static_cast<uint32_t>(currentShadingModel)] =
            static_cast<half>(static_cast<float>(histogram->shadingModel_[static_cast<uint32_t>(currentShadingModel)]) + affixAlpha);
    }
    
    FUNCTION_MODIFIER_INLINE void RecordScreenGBuffer(const TraceContext& traceContext, denoise::ScreenGBuffer* __restrict__ histogram = nullptr)
    {
        if(!histogram)
        {
            return;
        }
        if(traceContext.intersectionContext_.bHit_)
        {
            if(histogram->validFrameIndex_ == 0)
            {
                histogram->albedoX_ = 0.0f;
                histogram->albedoY_ = 0.0f;
                histogram->albedoZ_ = 0.0f;
                
                histogram->normalX_ = 0.0f;
                histogram->normalY_ = 0.0f;
                histogram->normalZ_ = 0.0f;
                
                histogram->metallic_ = 0.0f;
                histogram->roughness_ = 0.0f;
                
                histogram->depth_ = 0.0f;
            }
            const float affixAlpha = 1.0f / static_cast<float>(histogram->validFrameIndex_ + 1ull);
            const float preAlpha = static_cast<float>(histogram->validFrameIndex_) / static_cast<float>(histogram->validFrameIndex_ + 1ull);
            histogram->albedoX_ = static_cast<half>(static_cast<float>(histogram->albedoX_) * preAlpha);
            histogram->albedoY_ = static_cast<half>(static_cast<float>(histogram->albedoY_) * preAlpha);
            histogram->albedoZ_ = static_cast<half>(static_cast<float>(histogram->albedoZ_) * preAlpha);
            
            histogram->normalX_ = static_cast<half>(static_cast<float>(histogram->normalX_) * preAlpha);
            histogram->normalY_ = static_cast<half>(static_cast<float>(histogram->normalY_) * preAlpha);
            histogram->normalZ_ = static_cast<half>(static_cast<float>(histogram->normalZ_) * preAlpha);
            
            histogram->metallic_ = static_cast<half>(static_cast<float>(histogram->metallic_) * preAlpha);
            histogram->roughness_ = static_cast<half>(static_cast<float>(histogram->roughness_) * preAlpha);
            histogram->depth_ = histogram->depth_ * preAlpha;

            histogram->albedoX_ = static_cast<half>(static_cast<float>(histogram->albedoX_) + affixAlpha * traceContext.intersectionContext_.gbuffer_.albedo_.x);
            histogram->albedoY_ = static_cast<half>(static_cast<float>(histogram->albedoY_) + affixAlpha * traceContext.intersectionContext_.gbuffer_.albedo_.y);
            histogram->albedoZ_ = static_cast<half>(static_cast<float>(histogram->albedoZ_) + affixAlpha * traceContext.intersectionContext_.gbuffer_.albedo_.z);
            
            histogram->normalX_ = static_cast<half>(static_cast<float>(histogram->normalX_) + affixAlpha * traceContext.intersectionContext_.hitNormal_.x);
            histogram->normalY_ = static_cast<half>(static_cast<float>(histogram->normalY_) + affixAlpha * traceContext.intersectionContext_.hitNormal_.y);
            histogram->normalZ_ = static_cast<half>(static_cast<float>(histogram->normalZ_) + affixAlpha * traceContext.intersectionContext_.hitNormal_.z);
            
            histogram->metallic_ = static_cast<half>(static_cast<float>(histogram->metallic_) + affixAlpha * traceContext.intersectionContext_.gbuffer_.metallic_);
            histogram->roughness_ = static_cast<half>(static_cast<float>(histogram->roughness_) + affixAlpha * traceContext.intersectionContext_.gbuffer_.roughness_);
            histogram->depth_ = histogram->depth_ + affixAlpha * traceContext.intersectionContext_.distance_;
            
            histogram->validFrameIndex_++;
            RecordShadingModel(traceContext.shadingModel_, histogram);
        }
        else
        {
            RecordShadingModel(EShadingModel::MAT_MAX, histogram);
        }
        histogram->frameIndex_++;
    }

    // Not Using
    FUNCTION_MODIFIER_INLINE void ShadingMax(const ScreenGBuffer& g, float& pmax, unsigned char& argmax)
    {
        float best = -1.f;
        int idx = 0;
        CUDA_UNROLL
        for (int i=0; i<MAX_SHADING_MODEL+1; ++i)
        {
            float p = static_cast<float>(g.shadingModel_[i]);
            if (p>best)
            {
                best=p;
                idx=i;
            }
        }
        pmax = fmaxf(best, 0.f);
        argmax = static_cast<unsigned char>(idx);
    }
    
    FUNCTION_MODIFIER_INLINE
    float ComputeSVGFWeight(const ScreenGBuffer& a, const ScreenGBuffer& b)
    {
        float wTerm = 0.f;
        {
            const float3 ax = float3{static_cast<float>(a.albedoX_), static_cast<float>(a.albedoY_), static_cast<float>(a.albedoZ_)};
            const float3 bx = float3{static_cast<float>(b.albedoX_), static_cast<float>(b.albedoY_), static_cast<float>(b.albedoZ_)};
            const float3 d1 = (ax - bx);
            const float d2 = dot(d1, d1);
            wTerm += d2 * FilterParams::inv2Sigma2Albedo_;
        }
        {
            const float3 na = normalize(float3{static_cast<float>(a.normalX_), static_cast<float>(a.normalY_), static_cast<float>(a.normalZ_)});
            const float3 nb = normalize(float3{static_cast<float>(b.normalX_), static_cast<float>(b.normalY_), static_cast<float>(b.normalZ_)});
            const float dotV = dot(na, nb);
            const float d = 1.0f - dotV; // [0, 1]
            wTerm += d*d * FilterParams::inv2Sigma2Normal_;
        }
        {
            //const float d = (1.0f / (a.depth_ + 1.0f)) - (1.0f / (b.depth_ + 1.0f));
            const float denom = fmaxf(fmaxf(a.depth_, b.depth_), 1e-6f);
            const float d = (a.depth_ - b.depth_) / denom;
            wTerm += d * d * FilterParams::inv2Sigma2Depth_;
        }
        {
#ifdef __CUDA_ARCH__
            const float dm = static_cast<float>(a.metallic_ - b.metallic_);
#else
            const float dm = static_cast<float>(a.metallic_) - static_cast<float>(b.metallic_);
#endif
            wTerm += dm * dm * FilterParams::inv2Sigma2Metallic_;
        }
        {
#ifdef __CUDA_ARCH__
            const float dr = static_cast<float>(a.roughness_ - b.roughness_);
#else
            const float dr = static_cast<float>(a.roughness_) - static_cast<float>(b.roughness_);
#endif
            wTerm += dr * dr * FilterParams::inv2Sigma2Rough_;
        }
        {
            float l1 = 0.f;
            CUDA_UNROLL
            for (int i = 0; i < MAX_SHADING_MODEL+1; ++i)
            {
#ifdef __CUDA_ARCH__
                l1 += fabsf(static_cast<float>(a.shadingModel_[i] - b.shadingModel_[i]));
#else
                l1 += fabsf(static_cast<float>(a.shadingModel_[i]) - static_cast<float>(b.shadingModel_[i]));
#endif
            }
            const float tv = 0.5f * l1;
            wTerm += tv * tv * FilterParams::inv2Sigma2ShadingModel_;
        }
        const float w = expf(-wTerm);
        return w;
    }

    static constexpr float FILTER_ALBEDO_EPS = 0.1f;
    FUNCTION_MODIFIER_INLINE
    float3 SeparateSample(const ScreenGBuffer& buffer, const float3 color)
    {
        //return color;
        const float3 sample = log(color + 1.0f);
        return sample;
        //const float3 albedo = {static_cast<float>(buffer.albedoX_), static_cast<float>(buffer.albedoY_), static_cast<float>(buffer.albedoZ_)};
        //const float3 sample = log(color / (albedo + FILTER_ALBEDO_EPS) + 1.0f);
        //return sample;
    }

    FUNCTION_MODIFIER_INLINE
    float3 CompositeSample(const ScreenGBuffer& buffer, const float3 sample)
    {
        //return sample;
        const float3 color = max(exp(sample) - 1.0f, 0.0f);
        return color;
        //const float3 albedo = {static_cast<float>(buffer.albedoX_), static_cast<float>(buffer.albedoY_), static_cast<float>(buffer.albedoZ_)};
        //const float3 color = max(exp(sample) - 1.0f, 0.0f) * (albedo + FILTER_ALBEDO_EPS);
        //return color;
    }
    struct TStatistic
    {
        float varianceBesselCorrectedX_ = 0.0f;
        float varianceBesselCorrectedY_ = 0.0f;
        float varianceBesselCorrectedZ_ = 0.0f;
        float varianceThetaHatX_ = 0.0f;
        float varianceThetaHatY_ = 0.0f;
        float varianceThetaHatZ_ = 0.0f;
        float thetaX_ = 0.0f; // miu + M3 / (6 sigmaPow2 num)
        float thetaY_ = 0.0f;
        float thetaZ_ = 0.0f;
    };
    
    FUNCTION_MODIFIER_INLINE
    TStatistic ComputeTStatistic(const ScreenStatisticsBuffer& stat)
    {
        // M3DivNum_: 1
        TStatistic result;
        const double num = static_cast<double>(stat.num_);
        const double invNumMinus1 = 1.0 / static_cast<double>(max(stat.num_ - 1u, 1u));
        const double M2_x = static_cast<double>(stat.M2DivNum_.x) * num; // N
        const double M2_y = static_cast<double>(stat.M2DivNum_.y) * num;
        const double M2_z = static_cast<double>(stat.M2DivNum_.z) * num;
        result.varianceBesselCorrectedX_ = static_cast<float>(M2_x * invNumMinus1); // 1
        result.varianceBesselCorrectedY_ = static_cast<float>(M2_y * invNumMinus1);
        result.varianceBesselCorrectedZ_ = static_cast<float>(M2_z * invNumMinus1);
        result.varianceThetaHatX_ = static_cast<float>(result.varianceBesselCorrectedX_ / num); // 1 / N
        result.varianceThetaHatY_ = static_cast<float>(result.varianceBesselCorrectedY_ / num);
        result.varianceThetaHatZ_ = static_cast<float>(result.varianceBesselCorrectedZ_ / num);
        // M3_i / (6 * sig_i^2 * n_i)
        // (M3_i / n_i) / (6 * sig_i^2) = M3DivNum / 6 / sig_i^2
        result.thetaX_ = stat.mean_.x + stat.M3DivNum_.x / 6.0f / result.varianceBesselCorrectedX_; // miu + M3 / (6 sigmaPow2 num)
        result.thetaY_ = stat.mean_.y + stat.M3DivNum_.y / 6.0f / result.varianceBesselCorrectedY_;
        result.thetaZ_ = stat.mean_.z + stat.M3DivNum_.z / 6.0f / result.varianceBesselCorrectedZ_;
        return result;
    }

    FUNCTION_MODIFIER_INLINE
    float ComputeTStatisticAsymW(const float varianceThetaHatA, const float thetaA, const float varianceThetaHatB, const float thetaB)
    {
        const float t = (thetaA - thetaB);
        const float t2 = t * t;
        const float tUpper = t2 + varianceThetaHatB;
        const float tLower = t2 + varianceThetaHatA + varianceThetaHatB;
        return tUpper / fmaxf(tLower, 1.0e-6f);
    }

    FUNCTION_MODIFIER_INLINE
    float3 ComputeTStatisticAsymW(const TStatistic& A, const TStatistic& B)
    {
        const float x = ComputeTStatisticAsymW(A.varianceThetaHatX_, A.thetaX_, B.varianceThetaHatX_, B.thetaX_);
        const float y = ComputeTStatisticAsymW(A.varianceThetaHatY_, A.thetaY_, B.varianceThetaHatY_, B.thetaY_);
        const float z = ComputeTStatisticAsymW(A.varianceThetaHatZ_, A.thetaZ_, B.varianceThetaHatZ_, B.thetaZ_);
        return {x, y, z};
    }

    FUNCTION_MODIFIER_INLINE
    float3 ComputeTStatisticAsymT(const TStatistic& A, const TStatistic& B)
    {
        const float3 AsymW = ComputeTStatisticAsymW(A, B);
        return sqrt(float3{0.5f, 0.5f, 0.5f} / (float3{1.0f, 1.0f, 1.0f} - AsymW) - 1.0f);
    }

    FUNCTION_MODIFIER_INLINE
    float ComputeWelchNu(
        const float varianceBesselCorrectedA, const uint32_t numA,
        const float varianceBesselCorrectedB, const uint32_t numB)
    {
        // varianceBesselCorrectedA: s^2
        const float upperA = varianceBesselCorrectedA / static_cast<float>(numA);
        const float upperB = varianceBesselCorrectedB / static_cast<float>(numB);
        const float upper = upperA + upperB;
        const float upper2 = upper * upper;
        const float lowerA = upperA * upperA / static_cast<float>(max(numA - 1u, 1u));
        const float lowerB = upperB * upperB / static_cast<float>(max(numB - 1u, 1u));
        return upper2 / (lowerA + lowerB);
    }
    
    FUNCTION_MODIFIER_INLINE
    float ComputeTStatisticFinal(const ScreenStatisticsBuffer& A, const ScreenStatisticsBuffer& B)
    {
        const TStatistic tA = ComputeTStatistic(A);
        const TStatistic tB = ComputeTStatistic(B);
        const float3 T = ComputeTStatisticAsymT(tA, tB);
        const float nuX = ComputeWelchNu(tA.varianceBesselCorrectedX_, A.num_, tB.varianceBesselCorrectedX_, B.num_);
        const float nuY = ComputeWelchNu(tA.varianceBesselCorrectedY_, A.num_, tB.varianceBesselCorrectedY_, B.num_);
        const float nuZ = ComputeWelchNu(tA.varianceBesselCorrectedZ_, A.num_, tB.varianceBesselCorrectedZ_, B.num_);
        const float tX = quantile::LookupTQuantile005(floorToInt(nuX));
        const float tY = quantile::LookupTQuantile005(floorToInt(nuY));
        const float tZ = quantile::LookupTQuantile005(floorToInt(nuZ));
        return T.x < tX && T.y < tY && T.z < tZ ? 1.0f : 0.0f;
    }
    
    template<int RADIUS = 7>
    __global__ void Bilateral16x16(
        const float3* __restrict__ inHDR,
        float3* __restrict__ outHDR,
        const ScreenGBuffer* __restrict__ screenGBuffer,
        const ScreenStatisticsBuffer* __restrict__ screenStatisticsBuffer,
        const int2 size)
    {
        constexpr int BX = 16, BY = 16;
        constexpr int SX = BX + 2 * RADIUS;
        constexpr int SY = BY + 2 * RADIUS;

        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int ox = bx * BX + tx;
        const int oy = by * BY + ty;
        const bool inRange = (ox < size.x) && (oy < size.y);

        __shared__ float3 sharedHDR[SY][SX];

        const int gx0 = bx * BX - RADIUS;
        const int gy0 = by * BY - RADIUS;

        CUDA_UNROLL
        for (int sy = ty; sy < SY; sy += BY)
        {
            const int gy = clampi(gy0 + sy, 0, size.y - 1);
            const int row = gy * size.x;
            CUDA_UNROLL
            for (int sx = tx; sx < SX; sx += BX)
            {
                const int gx  = clampi(gx0 + sx, 0, size.x - 1);
                sharedHDR[sy][sx] = inHDR[row + gx];
            }
        }
        __syncthreads();
        if (!inRange)
        {
            return;
        }
        const int CX = tx + RADIUS;
        const int CY = ty + RADIUS;

        const int cgy  = clampi(gy0 + CY, 0, size.y - 1);
        const int cgx  = clampi(gx0 + CX, 0, size.x - 1);
        const int cidx = cgy * size.x + cgx;

        const ScreenGBuffer centerScreenGBuffer = screenGBuffer[cidx];
        const ScreenStatisticsBuffer centerScreenStatisticsBuffer = screenStatisticsBuffer[cidx];

        float3 accumulatedColor = make_float3(0,0,0);
        float accumulatedWeight = 0.0f;

        constexpr float CENTER_PIXEL_EPS = 1.0e-6f;
        accumulatedColor = accumulatedColor + SeparateSample(centerScreenGBuffer, sharedHDR[CY][CX]) * CENTER_PIXEL_EPS;
        accumulatedWeight = accumulatedWeight + CENTER_PIXEL_EPS;

        CUDA_UNROLL
        for (int j = -RADIUS; j <= RADIUS; ++j)
        {
            const int SYj = CY + j;
            const int ngy = clampi(gy0 + SYj, 0, size.y - 1);
            const int nrow = ngy * size.x;
            CUDA_UNROLL
            for (int i = -RADIUS; i <= RADIUS; ++i)
            {
                const int SXi = CX + i;
                const int ngx = clampi(gx0 + SXi, 0, size.x - 1);
                const ScreenGBuffer currentScreenGBuffer = screenGBuffer[nrow + ngx];
                const ScreenStatisticsBuffer currentScreenStatisticsBuffer = screenStatisticsBuffer[nrow + ngx];

                const float weight =
                    ComputeSVGFWeight(centerScreenGBuffer, currentScreenGBuffer) *
                    expf(-(i*i + j*j) * FilterParams::inv2Sigma2Spatial_) *
                    ComputeTStatisticFinal(centerScreenStatisticsBuffer, currentScreenStatisticsBuffer);
                
                accumulatedColor = accumulatedColor + SeparateSample(currentScreenGBuffer, sharedHDR[SYj][SXi]) * weight;
                accumulatedWeight = accumulatedWeight + weight;
            }
        }
        outHDR[oy * size.x + ox] = CompositeSample(centerScreenGBuffer, accumulatedColor / accumulatedWeight);
    }
}