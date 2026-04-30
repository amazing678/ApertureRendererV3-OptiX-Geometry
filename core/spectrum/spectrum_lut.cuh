#pragma once
#include <cuda_runtime.h>
#include <omp.h>

#include "vector.cuh"
#include "spectrum_basis.cuh"
#include "sample_ciexyz.cuh"
#include "tonemap.cuh"

namespace spectrum
{
    static __global__ void PrecomputeRGBLUTKernel(float* __restrict__ lut)
    {
        constexpr int RES = SPECTRUM_RGB_LUT_RES;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= RES || y >= RES || z >= RES)
        {
            return;
        }
        const int voxel = (z * RES + y) * RES + x;
        constexpr float invRes = 1.0f / static_cast<float>(RES);

        float in3[3];
        in3[0] = (x + 0.5f) * invRes;  // R in [0,1]
        in3[1] = (y + 0.5f) * invRes;  // G in [0,1]
        in3[2] = (z + 0.5f) * invRes;  // B in [0,1]

        float outK[spectrum::query::KERNEL];
        spectrum::query::RGBEncode(in3, outK);

        #pragma unroll
        for (int k = 0; k < spectrum::query::KERNEL; ++k)
        {
            lut[voxel * spectrum::query::KERNEL + k] = outK[k];
        }
    }
    static __global__ void PrecomputeLambdaLUTKernel(float* __restrict__ lut)
    {
        constexpr int RES = SPECTRUM_LAMBDA_LUT_RES;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= RES)
        {
            return;
        }
        constexpr float lamMin = 360.0f;
        constexpr float lamMax = 780.0f;
        const float step = (lamMax - lamMin) / static_cast<float>(RES);
        const float lambda = lamMin + (i + 0.5f) * step;

        const float in1[1] = { lambda };
        float outK[spectrum::query::KERNEL];
        spectrum::query::LambdaEncode(in1, outK);
#pragma unroll
        for (int k = 0; k < spectrum::query::KERNEL; ++k)
        {
            lut[i * spectrum::query::KERNEL + k] = outK[k];
        }
    }
    // host
    inline void PrecomputeSpectrumLUTsToDevice(
        float* __restrict__ rgbLUT, // size: SPECTRUM_RGB_LUT_RES^3 * KERNEL_N
        float* __restrict__ lambdaLUT, // size: SPECTRUM_LAMBDA_LUT_RES * KERNEL_N
        cudaStream_t stream = 0
    )
    {
        dim3 block3(8, 8, 8);
        dim3 grid3((SPECTRUM_RGB_LUT_RES + block3.x - 1) / block3.x,
                   (SPECTRUM_RGB_LUT_RES + block3.y - 1) / block3.y,
                   (SPECTRUM_RGB_LUT_RES + block3.z - 1) / block3.z);
        PrecomputeRGBLUTKernel<<<grid3, block3, 0, stream>>>(rgbLUT);
        constexpr int tpb = 256;
        constexpr int g1 = (SPECTRUM_LAMBDA_LUT_RES + tpb - 1) / tpb;
        PrecomputeLambdaLUTKernel<<<g1, tpb, 0, stream>>>(lambdaLUT);
        CHECK_ERROR();
        return;
    }

    // device
    namespace lut
    {
        static_assert(spectrum::query::KERNEL % 4 == 0);
        FUNCTION_MODIFIER_INLINE void __zeroK(float* __restrict__ outK)
        {
            CUDA_UNROLL
            for (int k = 0; k < spectrum::query::KERNEL / 4; ++k)
            {
                (reinterpret_cast<float4*>(outK))[k] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        FUNCTION_MODIFIER_INLINE void __linear_coord(float g, int res, int& i0, int& i1, float& t)
        {
            const float fi0 = floorf(g);
            i0 = static_cast<int>(fi0);
            i1 = i0 + 1;
            if (i1 >= res)
            {
                i1 = res - 1;
            }
            t = g - fi0;
            if (i0 == i1)
            {
                t = 0.0f;
            }
        }
        FUNCTION_MODIFIER_INLINE void __accum_vec4(const float* __restrict__ base, float w, float* __restrict__ outK)
        {
            constexpr int K4 = spectrum::query::KERNEL / 4;
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(base);
            
            CUDA_UNROLL
            for (int q = 0; q < K4; ++q)
            {
                float4 v = p4[q];
                const int o = q * 4;
                outK[o+0] = fmaf(w, v.x, outK[o+0]);
                outK[o+1] = fmaf(w, v.y, outK[o+1]);
                outK[o+2] = fmaf(w, v.z, outK[o+2]);
                outK[o+3] = fmaf(w, v.w, outK[o+3]);
            }
        }

        FUNCTION_MODIFIER_INLINE SpectrumBasis QueryRGBBasis(const float3 rgb, const float* __restrict__ lutRGB)
        {
            SpectrumBasis result = {};
            constexpr int RES = SPECTRUM_RGB_LUT_RES;
            float gx = rgb.x * static_cast<float>(RES) - 0.5f;
            float gy = rgb.y * static_cast<float>(RES) - 0.5f;
            float gz = rgb.z * static_cast<float>(RES) - 0.5f;
            const float maxC = static_cast<float>(RES) - 1.0f;
            gx = fminf(fmaxf(gx, 0.0f), maxC);
            gy = fminf(fmaxf(gy, 0.0f), maxC);
            gz = fminf(fmaxf(gz, 0.0f), maxC);

            int x0, x1, y0, y1, z0, z1;
            float tx, ty, tz;
            __linear_coord(gx, RES, x0, x1, tx);
            __linear_coord(gy, RES, y0, y1, ty);
            __linear_coord(gz, RES, z0, z1, tz);

            const float wx0 = 1.0f - tx;
            const float wx1 = tx;
            const float wy0 = 1.0f - ty;
            const float wy1 = ty;
            const float wz0 = 1.0f - tz;
            const float wz1 = tz;

            const float w000 = wx0 * wy0 * wz0;
            const float w100 = wx1 * wy0 * wz0;
            const float w010 = wx0 * wy1 * wz0;
            const float w110 = wx1 * wy1 * wz0;
            const float w001 = wx0 * wy0 * wz1;
            const float w101 = wx1 * wy0 * wz1;
            const float w011 = wx0 * wy1 * wz1;
            const float w111 = wx1 * wy1 * wz1;

            constexpr int sx = spectrum::query::KERNEL;
            constexpr int sy = RES * sx;
            constexpr int sz = RES * sy;
            const float* b000 = lutRGB + (x0*sx + y0*sy + z0*sz);
            const float* b100 = lutRGB + (x1*sx + y0*sy + z0*sz);
            const float* b010 = lutRGB + (x0*sx + y1*sy + z0*sz);
            const float* b110 = lutRGB + (x1*sx + y1*sy + z0*sz);
            const float* b001 = lutRGB + (x0*sx + y0*sy + z1*sz);
            const float* b101 = lutRGB + (x1*sx + y0*sy + z1*sz);
            const float* b011 = lutRGB + (x0*sx + y1*sy + z1*sz);
            const float* b111 = lutRGB + (x1*sx + y1*sy + z1*sz);

            __accum_vec4(b000, w000, result.spectrumBasis_);
            __accum_vec4(b100, w100, result.spectrumBasis_);
            __accum_vec4(b010, w010, result.spectrumBasis_);
            __accum_vec4(b110, w110, result.spectrumBasis_);
            __accum_vec4(b001, w001, result.spectrumBasis_);
            __accum_vec4(b101, w101, result.spectrumBasis_);
            __accum_vec4(b011, w011, result.spectrumBasis_);
            __accum_vec4(b111, w111, result.spectrumBasis_);
            return result;
        }

        FUNCTION_MODIFIER_INLINE SpectrumBasis QueryLambda(float lambdaNm, const float* __restrict__ lutLam)
        {
            SpectrumBasis result = {};
            constexpr int RES = SPECTRUM_LAMBDA_LUT_RES;
            constexpr float lamMin = 360.0f;
            constexpr float lamMax = 780.0f;
            float u = (lambdaNm - lamMin) / (lamMax - lamMin);
            u = fminf(fmaxf(u, 0.0f), 1.0f);

            float g = u * static_cast<float>(RES) - 0.5f;
            constexpr float maxG = static_cast<float>(RES) - 1.0f;
            g = fminf(fmaxf(g, 0.0f), maxG);

            int i0, i1;
            float t;
            __linear_coord(g, RES, i0, i1, t);
            const float w0 = 1.0f - t;
            const float w1 = t;

            constexpr int K = spectrum::query::KERNEL;
            const float* b0 = lutLam + i0 * K;
            const float* b1 = lutLam + i1 * K;

            constexpr int K4 = spectrum::query::KERNEL / 4;
            const float4* __restrict__ p0 = reinterpret_cast<const float4*>(b0);
            const float4* __restrict__ p1 = reinterpret_cast<const float4*>(b1);
            
            CUDA_UNROLL
            for (int q = 0; q < K4; ++q)
            {
                const float4 a = p0[q];
                const float4 b = p1[q];
                const int o = q * 4;
                result.spectrumBasis_[o+0] = a.x * w0 + b.x * w1;
                result.spectrumBasis_[o+1] = a.y * w0 + b.y * w1;
                result.spectrumBasis_[o+2] = a.z * w0 + b.z * w1;
                result.spectrumBasis_[o+3] = a.w * w0 + b.w * w1;
            }
            return result;
        }

        FUNCTION_MODIFIER_INLINE float Interact(const SpectrumBasis& lambdaBasis, const float3 RGB, const float* __restrict__ lutRGB)
        {
            const float luminance = max3(RGB);
            if(luminance > 1.0f)
            {
                const SpectrumBasis rgbBasis = QueryRGBBasis(RGB / luminance, lutRGB);
                return rgbBasis.query(lambdaBasis) * luminance;
            }
            else
            {
                const SpectrumBasis rgbBasis = QueryRGBBasis(RGB, lutRGB);
                return rgbBasis.query(lambdaBasis);
            }
        }
    }

    inline void ValidationSpectrum()
    {
        const auto integral = [](const float3 color)
        {
            float RGBBasis[spectrum::query::KERNEL];
            spectrum::query::RGBEncode(&color.x, RGBBasis);

            const auto RNG01 = []() -> double
            {
                thread_local uint64_t s =
                    0x9E3779B97F4A7C15ull ^
                    (0xBF58476D1CE4E5B9ull * static_cast<uint64_t>(omp_get_thread_num()));
                s = s * 2862933555777941757ull + 3037000493ull;
                return ((s >> 11) * (1.0 / 9007199254740992.0));
            };
            const auto RNG01F = [&RNG01]()->float
            {
                return static_cast<float>(RNG01());
            };

            float3 xyz = {0.0f, 0.0f, 0.0f};
            float3 xyzSimple = {0.0f, 0.0f, 0.0f};
            constexpr int N = 10000;

            #pragma omp parallel
            {
                float3 xyzLocal = {0.0f, 0.0f, 0.0f};
                float3 xyzSimpleLocal = {0.0f, 0.0f, 0.0f};
                #pragma omp for nowait
                for(int i=0; i<N; i++)
                {
                    float lambdaBasis[spectrum::query::KERNEL];
                    float lambdaBasisSimple[spectrum::query::KERNEL];
                    const spectrum::sample::SpectrumSample lambdaSample =
                        spectrum::sample::SampleCIE1931({RNG01F(), RNG01F(), RNG01F()});
                    const spectrum::sample::SpectrumSample lambdaSampleSimple =
                        spectrum::sample::SampleCIE1931_Simple({RNG01F(), RNG01F(), RNG01F()});

                    spectrum::query::LambdaEncode(&lambdaSample.lambda_, lambdaBasis);
                    spectrum::query::LambdaEncode(&lambdaSampleSimple.lambda_, lambdaBasisSimple);

                    const float strength =
                        spectrum::query::SpectralInteract(RGBBasis, lambdaBasis);
                    const float strengthSimple =
                        spectrum::query::SpectralInteract(RGBBasis, lambdaBasisSimple);

                    xyzLocal = xyzLocal
                        + strength
                            * spectrum::LambdaToCIE1931_XYZ(lambdaSample.lambda_)
                            * spectrum::D65Norm(lambdaSample.lambda_)
                            / lambdaSample.pdf_;

                    xyzSimpleLocal = xyzSimpleLocal
                        + strengthSimple
                            * spectrum::LambdaToCIE1931_XYZ(lambdaSampleSimple.lambda_)
                            * spectrum::D65Norm(lambdaSampleSimple.lambda_)
                            / lambdaSampleSimple.pdf_;
                }

                #pragma omp critical
                {
                    xyz = xyz + xyzLocal;
                    xyzSimple = xyzSimple + xyzSimpleLocal;
                }
            }
            xyz = xyz / static_cast<float>(N);
            xyzSimple = xyzSimple / static_cast<float>(N);

            const float3 rgb = XYZ2SRGBLinearD65(xyz);
            const float3 rgbSimple = XYZ2SRGBLinearD65(xyzSimple);
            printf("-----------------------------\n");
            printf("color    : {%.8f, %.8f, %.8f}\n", color.x, color.y, color.z);
            printf("xyz      : {%.8f, %.8f, %.8f}\n", xyz.x, xyz.y, xyz.z);
            printf("xyzSimple: {%.8f, %.8f, %.8f}\n", xyzSimple.x, xyzSimple.y, xyzSimple.z);
            printf("rgb      : {%.8f, %.8f, %.8f}\n", rgb.x, rgb.y, rgb.z);
            printf("rgbSimple: {%.8f, %.8f, %.8f}\n", rgbSimple.x, rgbSimple.y, rgbSimple.z);
            printf("-----------------------------\n");
        };

        integral(color::White());
        integral(color::Red());
        integral(color::Green());
        integral(color::Blue());
    }
}
