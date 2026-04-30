#pragma once

#include <functional>
#include <cuda_runtime.h>
#include <mma.h>

#include "vector.cuh"
#include <mma.h>
#include <cuda_fp16.h>

#ifndef __CUDA_ARCH__
#ifndef __fmaf_rn
#define __fmaf_rn(a,b,c) fmaf((a),(b),(c))
#endif
#endif

namespace tensor
{
    template<int Count>
    FUNCTION_MODIFIER_INLINE void Copy(const float* __restrict__ src, float* __restrict__ dst)
    {
        static_assert(Count % 4 == 0, "Count must be a multiple of 4");
#ifdef __CUDA_ARCH__
        // assert((((uintptr_t)src | (uintptr_t)dst) & 0xF) == 0 && "need 16B alignment");
        constexpr int N4 = Count / 4;
        const float4* __restrict__ s4 = reinterpret_cast<const V*>(src);
        float4* __restrict__ d4 = reinterpret_cast<V*>(dst);
        CUDA_UNROLL
        for (int i = 0; i < N4; ++i)
        {
            float4 v = s4[i];
            d4[i] = v;
        }
#else
        std::memcpy(dst, src, sizeof(float) * Count);
#endif
    }
    template<int Count>
FUNCTION_MODIFIER_INLINE void SetZero(float* __restrict__ dst)
    {
        static_assert(Count % 4 == 0, "Count must be a multiple of 4");
#ifdef __CUDA_ARCH__
        // assert((((uintptr_t)dst) & 0xF) == 0 && "need 16B alignment");
        constexpr int N4 = Count / 4;
        float4* __restrict__ d4 = reinterpret_cast<float4*>(dst);
        CUDA_UNROLL
        for (int i = 0; i < N4; ++i)
        {
            d4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#else
        std::memset(dst, 0, sizeof(float) * Count);
        #endif
    }
    // NOTICE: need 16B alignment!!!
    // NOTICE: in out must not overlap!!!
    template<int From, int To, bool bBias, bool bReLU>
    FUNCTION_MODIFIER_INLINE void LinearRowMajor_Legacy(
    const float* __restrict__ in,
    float* __restrict__ out,
    int readAt, int writeAt,
    const float* __restrict__ weight, // layout: i*from + j
    const float* __restrict__ bias)
    {
        CUDA_UNROLL
        for (int i = 0; i < To; ++i)
        {
            const float* __restrict__ w = weight + i * From;
            const float* __restrict__ x = in + readAt;
            float acc = 0.0f;
            if constexpr (bBias)
            {
                acc = bias[i];
            }
            int j = 0;
            const int limit = From & ~3;
            for (; j < limit; j += 4)
            {
                const float4 vX = *reinterpret_cast<const float4*>(x + j);
                const float4 vW = *reinterpret_cast<const float4*>(w + j);
                acc = __fmaf_rn(vX.x, vW.x, acc);
                acc = __fmaf_rn(vX.y, vW.y, acc);
                acc = __fmaf_rn(vX.z, vW.z, acc);
                acc = __fmaf_rn(vX.w, vW.w, acc);
            }
            for (; j < From; ++j)
            {
                acc = __fmaf_rn(x[j], w[j], acc);
            }
            if constexpr (bReLU)
            {
                out[writeAt + i] = fmaxf(acc, 0.0f);
            }
            else
            {
                out[writeAt + i] = acc;
            }
        }
    }

    // NOTICE: need 16B alignment!!! 
    // NOTICE: in out must not overlap!!!
    template<int From, int To, bool bBias, bool bReLU>
FUNCTION_MODIFIER_INLINE void LinearRowMajor_Opt(
    const float* __restrict__ in,
    float* __restrict__ out,
    int readAt, int writeAt,
    const float* __restrict__ weight, // layout: i*From + j
    const float* __restrict__ bias)
    {
        const float* __restrict__ xBase = in + readAt;
        float acc[To];
        CUDA_UNROLL
        for (int i = 0; i < To; ++i)
        {
            if constexpr (bBias)
            {
                acc[i] = bias[i];
            }
            else
            {
                acc[i] = 0.0f;
            }
        }
        int j = 0;
        const uintptr_t addr = reinterpret_cast<uintptr_t>(xBase);
        const int mis = (4 - ((addr >> 2) & 0x3)) & 0x3;  // 0..3
        const int prologue = (mis <= From) ? mis : From;
        for (; j < prologue; ++j)
        {
            float xj = xBase[j];
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                acc[i] = __fmaf_rn(xj, weight[i*From + j], acc[i]);
            }
        }
        const int limit = ((From - j) & ~3) + j;
        for (; j < limit; j += 4)
        {
            const float4 vx = *reinterpret_cast<const float4*>(xBase + j);
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                const float4 vw = *reinterpret_cast<const float4*>(weight + i*From + j);
                float a = acc[i];
                a = __fmaf_rn(vx.x, vw.x, a);
                a = __fmaf_rn(vx.y, vw.y, a);
                a = __fmaf_rn(vx.z, vw.z, a);
                a = __fmaf_rn(vx.w, vw.w, a);
                acc[i] = a;
            }
        }
        for (; j < From; ++j)
        {
            float xj = xBase[j];
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                acc[i] = __fmaf_rn(xj, weight[i*From + j], acc[i]);
            }
        }
        if constexpr (bReLU)
        {
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                out[writeAt + i] = fmaxf(acc[i], 0.0f);
            }
        }
        else
        {
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                out[writeAt + i] = acc[i];
            }
        }
    }
    template<int From, int To, bool bBias, bool bReLU>
    FUNCTION_MODIFIER_INLINE void LinearRowMajor_Fallback(
        const float* __restrict__ in,
        float* __restrict__ out,
        int readAt, int writeAt,
        const float* __restrict__ weight, // layout: i*From + j
        const float* __restrict__ bias)
    {
        const float* __restrict__ xBase = in + readAt;
        float acc[To];
        CUDA_UNROLL
        for (int i = 0; i < To; ++i)
        {
            if constexpr (bBias)
            {
                acc[i] = bias[i];
            }
            else
            {
                acc[i] = 0.0f;
            }
        }
        CUDA_UNROLL
        for (int j = 0; j < From; ++j)
        {
            const float xj = xBase[j];
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                acc[i] = __fmaf_rn(xj, weight[i*From + j], acc[i]);
            }
        }

        if constexpr (bReLU)
        {
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                out[writeAt + i] = fmaxf(acc[i], 0.0f);
            }
        }
        else
        {
            CUDA_UNROLL
            for (int i = 0; i < To; ++i)
            {
                out[writeAt + i] = acc[i];
            }
        }
    }
    
    // NOTICE: in out must not overlap!!!
    template<int From, int To, bool bBias, bool bReLU, bool Opt = true>
FUNCTION_MODIFIER_INLINE void LinearRowMajor(
    const float* __restrict__ in,
    float* __restrict__ out,
    int readAt, int writeAt,
    const float* __restrict__ weight, // layout: i*From + j
    const float* __restrict__ bias)
    {
        if constexpr (From < 4 || (From % 4) != 0)
        {
            LinearRowMajor_Fallback<From, To, bBias, bReLU>(in, out, readAt, writeAt, weight, bias);
        }
        else
        {
            static_assert((From % 4) == 0, "From must be multiple of 4 when using float4 loads.");
            if constexpr (Opt)
            {
                LinearRowMajor_Opt<From, To, bBias, bReLU>(in, out, readAt, writeAt, weight, bias);
            }
            else
            {
                LinearRowMajor_Legacy<From, To, bBias, bReLU>(in, out, readAt, writeAt, weight, bias);
            }
        }
    }
#ifdef TENSOR
    /*                            N
                                [][]
                 K              [][]        1             N
      [][][][][][][][][]        [][]        []        [][][][]
      [][][][][][][][][]        [][]        []        [][][][]
    M [][][][][][][][][]  x   K [][]   +  M []   =  M [][][][] 
      [][][][][][][][][]        [][]        []        [][][][]
      [][][][][][][][][]        [][]        []        [][][][]
                                [][]
                                [][]
    */
    template<int M, int K, int N = 32, int BaseDim = 16, typename MajorA = nvcuda::wmma::row_major, typename MajorB = nvcuda::wmma::row_major, typename MajorOut = nvcuda::wmma::row_major>
    FUNCTION_MODIFIER_DEVICE_INLINE
    void WMMA_ReluH(const half* __restrict__ A,
                    const half* __restrict__ B,
                    const half* __restrict__ C,
                    half* __restrict__ Y,
                    unsigned char* __restrict__ smemBase)
    {
        static_assert(BaseDim == 16, "BaseDim must be 16 for WMMA.");
        static_assert((M % BaseDim == 0) && (K % BaseDim == 0) && (N % BaseDim == 0), "M,K,N must be multiples of BaseDim.");

        constexpr int ldaA = std::is_same<MajorA, nvcuda::wmma::col_major>::value ? M : K;
        constexpr int ldbB = std::is_same<MajorB, nvcuda::wmma::col_major>::value ? K : N;
        constexpr int ldc  = std::is_same<MajorOut, nvcuda::wmma::col_major>::value ? M : N;

        int lane = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        const int warps = blockDim.x >> 5;

        float* warp_smem = reinterpret_cast<float*>(smemBase) + warp_id * (BaseDim * BaseDim + BaseDim);
        float* Ct = warp_smem;
        float* bias = warp_smem + BaseDim * BaseDim;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BaseDim, BaseDim, BaseDim, half, MajorA> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BaseDim, BaseDim, BaseDim, half, MajorB> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BaseDim, BaseDim, BaseDim, float> acc;

        const int tiles_x = N / BaseDim;
        const int tiles_y = M / BaseDim;
        const int warps_per_block = warps;
        const int warp_global = blockIdx.x * warps_per_block + warp_id;
        const int warp_stride = gridDim.x * warps_per_block;
        const int total_tiles = tiles_x * tiles_y;

        for (int tile = warp_global; tile < total_tiles; tile += warp_stride)
        {
            int tile_y = tile / tiles_x;
            int tile_x = tile % tiles_x;
            int i = tile_y * BaseDim;
            int j = tile_x * BaseDim;

            nvcuda::wmma::fill_fragment(acc, 0.0f);

            for (int kk = 0; kk < K; kk += BaseDim)
            {
                if constexpr (std::is_same<MajorA, nvcuda::wmma::col_major>::value)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag, A + i + kk * ldaA, ldaA);
                }
                else
                {
                    nvcuda::wmma::load_matrix_sync(a_frag, A + i * ldaA + kk, ldaA);
                }

                if constexpr (std::is_same<MajorB, nvcuda::wmma::col_major>::value)
                {
                    nvcuda::wmma::load_matrix_sync(b_frag, B + kk + j * ldbB, ldbB);
                }
                else
                {
                    nvcuda::wmma::load_matrix_sync(b_frag, B + kk * ldbB + j, ldbB);
                }

                nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            if (lane < BaseDim) bias[lane] = __half2float(C[i + lane]);
            __syncwarp();

            nvcuda::wmma::store_matrix_sync(Ct, acc, BaseDim, nvcuda::wmma::mem_row_major);
            __syncwarp();

            for (int t = lane; t < BaseDim * BaseDim; t += 32)
            {
                int r = t / BaseDim;
                int c = t % BaseDim;
                float v = Ct[t] + bias[r];
                v = v > 0.f ? v : 0.f;
                if constexpr (std::is_same<MajorOut, nvcuda::wmma::col_major>::value)
                {
                    Y[(i + r) + (j + c) * ldc] = __float2half_rn(v);
                }
                else
                {
                    Y[(i + r) * ldc + (j + c)] = __float2half_rn(v);
                }
            }
            __syncwarp();
        }
    }

    template<int M, int K, int N = 32, int BaseDim = 16, typename MajorA = nvcuda::wmma::row_major, typename MajorB = nvcuda::wmma::row_major, typename MajorOut = nvcuda::wmma::row_major>
    FUNCTION_MODIFIER_DEVICE
    void WMMA_ReluF(const float* __restrict__ A,
                    const float* __restrict__ B,
                    const float* __restrict__ C,
                    float* __restrict__ Y_f,
                    unsigned char* __restrict__ smemBase)
    {
        static_assert(BaseDim == 16, "BaseDim must be 16 for WMMA.");
        static_assert((M % BaseDim == 0) && (K % BaseDim == 0) && (N % BaseDim == 0), "M,K,N must be multiples of BaseDim.");

        constexpr int ldaA = std::is_same<MajorA, nvcuda::wmma::col_major>::value ? M : K;
        constexpr int ldbB = std::is_same<MajorB, nvcuda::wmma::col_major>::value ? K : N;
        constexpr int ldc  = std::is_same<MajorOut, nvcuda::wmma::col_major>::value ? M : N;

        int lane = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        const int warps = blockDim.x >> 5;

        auto align16 = [](size_t x){ return (x + 15) & ~static_cast<size_t>(15); };
        const size_t strideA = BaseDim * BaseDim * sizeof(half);
        const size_t strideB = BaseDim * BaseDim * sizeof(half);
        const size_t strideC = BaseDim * BaseDim * sizeof(float);
        [[maybe_unused]] const size_t strideBias = BaseDim * sizeof(float);

        const size_t ofsA = 0;
        const size_t ofsB = align16(ofsA + warps * strideA);
        const size_t ofsC = align16(ofsB + warps * strideB);
        const size_t ofsBias = align16(ofsC + warps * strideC);

        half*  As = reinterpret_cast<half*>(smemBase + ofsA)   + warp_id * (BaseDim * BaseDim);
        half*  Bs = reinterpret_cast<half*>(smemBase + ofsB)   + warp_id * (BaseDim * BaseDim);
        float* Ct = reinterpret_cast<float*>(smemBase + ofsC)  + warp_id * (BaseDim * BaseDim);
        float* bias = reinterpret_cast<float*>(smemBase + ofsBias) + warp_id * BaseDim;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BaseDim, BaseDim, BaseDim, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BaseDim, BaseDim, BaseDim, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BaseDim, BaseDim, BaseDim, float> acc;

        const int tiles_x = N / BaseDim;
        const int tiles_y = M / BaseDim;
        const int warps_per_block = warps;
        const int warp_global = blockIdx.x * warps_per_block + warp_id;
        const int warp_stride = gridDim.x * warps_per_block;
        const int total_tiles = tiles_x * tiles_y;

        for (int tile = warp_global; tile < total_tiles; tile += warp_stride)
        {
            int tile_y = tile / tiles_x;
            int tile_x = tile % tiles_x;
            int i = tile_y * BaseDim;
            int j = tile_x * BaseDim;

            nvcuda::wmma::fill_fragment(acc, 0.0f);

            for (int kk = 0; kk < K; kk += BaseDim)
            {
                for (int t = lane; t < BaseDim * BaseDim; t += 32)
                {
                    int r = t / BaseDim;
                    const int c = t % BaseDim;
                    int gr = i + r;
                    int gc = kk + c;
                    float a_val = std::is_same<MajorA, nvcuda::wmma::col_major>::value ? A[gr + gc * ldaA] : A[gr * ldaA + gc];
                    As[t] = __float2half_rn(a_val);
                }
                __syncwarp();

                for (int t = lane; t < BaseDim * BaseDim; t += 32)
                {
                    const int r = t / BaseDim;
                    int c = t % BaseDim;
                    int gr = kk + r;
                    int gc = j + c;
                    float b_val = std::is_same<MajorB, nvcuda::wmma::col_major>::value ? B[gr + gc * ldbB] : B[gr * ldbB + gc];
                    Bs[t] = __float2half_rn(b_val);
                }
                __syncwarp();

                nvcuda::wmma::load_matrix_sync(a_frag, As, BaseDim);
                nvcuda::wmma::load_matrix_sync(b_frag, Bs, BaseDim);
                nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            if (lane < BaseDim) bias[lane] = C[i + lane];
            __syncwarp();

            nvcuda::wmma::store_matrix_sync(Ct, acc, BaseDim, nvcuda::wmma::mem_row_major);
            __syncwarp();

            for (int t = lane; t < BaseDim * BaseDim; t += 32)
            {
                int r = t / BaseDim;
                int c = t % BaseDim;
                float v = Ct[t] + bias[r];
                v = v > 0.f ? v : 0.f;
                if constexpr (std::is_same<MajorOut, nvcuda::wmma::col_major>::value)
                {
                    Y_f[(i + r) + (j + c) * ldc] = v;
                }
                else
                {
                    Y_f[(i + r) * ldc + (j + c)] = v;
                }
            }
            __syncwarp();
        }
    }
    inline size_t GetWmmaReluFSmemBytes(int aWarpsPerBlock, int aBaseDim = 16)
    {
        auto align16 = [](size_t aX){ return (aX + 15) & ~static_cast<size_t>(15); };
        const size_t strideA   = aBaseDim * aBaseDim * sizeof(half); // 512
        const size_t strideB   = strideA; // 512
        const size_t strideC   = aBaseDim * aBaseDim * sizeof(float); // 1024
        const size_t strideBias= aBaseDim * sizeof(float); // 64

        const size_t ofsA    = 0;
        const size_t ofsB    = align16(ofsA + static_cast<size_t>(aWarpsPerBlock) * strideA);
        const size_t ofsC    = align16(ofsB + static_cast<size_t>(aWarpsPerBlock) * strideB);
        const size_t ofsBias = align16(ofsC + static_cast<size_t>(aWarpsPerBlock) * strideC);
        const size_t total   = align16(ofsBias + static_cast<size_t>(aWarpsPerBlock) * strideBias);
        return total;
    }
#endif
}