// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "random.cuh"
#include "guiding_bin.cuh"
#include "render/sample.cuh"
#include <cuda_runtime.h>
#include "octahedron.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/fill.h>

namespace pg
{
    struct PathGuidingSample
    {
#ifdef USE_3_CHANNEL_PROBE
        float3 radiance_ = float3{0.0f, 0.0f, 0.0f};
#else
        float radianceStrength_ = 0.0f;
#endif
        float3 radianceDirection_ = float3{0.0f, 0.0f, 0.0f};
        int4 voxelGrid_ = int4{0, 0, 0, 0}; // xyz, direction, use GetVoxelGrid to query
        bool valid_ = false;
    };

    // reduce helper
    struct IsSampleValid
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(const pg::PathGuidingSample& sample) const
        {
            return sample.valid_;
        }
    };
    struct SampleToVoxel
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE int4 operator()(const pg::PathGuidingSample& sample) const
        {
            return sample.voxelGrid_;
        }
    };
    struct Int4Less
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(const int4& a, const int4& b) const
        {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            if (a.z != b.z) return a.z < b.z;
            return a.w < b.w;
        }
    };
    struct Int4Equal
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(const int4& a, const int4& b) const
        {
            return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w;
        }
    };
    //
    // input: samples device ptr; N (total element count)
    // output: *validSamplePtr points all valid device ptr; 
    // return: valid element count
    [[nodiscard]] inline size_t FilterValidSamples(const pg::PathGuidingSample* samplesPtr, size_t N,
                                                   pg::PathGuidingSample* outValidSamplePtr,
                                                   cudaStream_t stream)
    {
        using namespace thrust;

        if (N == 0 || samplesPtr == nullptr || outValidSamplePtr == nullptr)
        {
            return 0;
        }

        const device_ptr<const pg::PathGuidingSample> begin(samplesPtr);
        const device_ptr<const pg::PathGuidingSample> end  = begin + N;

        const size_t M = thrust::count_if(cuda::par.on(stream), begin, end, IsSampleValid{});
        if (M == 0)
        {
            return 0;
        }
        const device_ptr<pg::PathGuidingSample> outBegin(outValidSamplePtr);
        const auto outEnd = thrust::copy_if(cuda::par.on(stream), begin, end, outBegin, IsSampleValid{});
        return static_cast<size_t>(outEnd - outBegin);
    }

    // input:  validSamplesPtr all valid; M: valid count
    // output: outVoxels,
    // return: unique VoxelGrid
    [[nodiscard]] inline size_t UniqueVoxelGridsFromValid(const pg::PathGuidingSample* validSamplesPtr, size_t M,
                                                          int4* outVoxelGrids,
                                                          cudaStream_t stream)
    {
        using namespace thrust;
        if (M == 0 || validSamplesPtr == nullptr || outVoxelGrids == nullptr)
        {
            return 0;
        }
        const device_ptr<const pg::PathGuidingSample> vBegin(validSamplesPtr);
        const device_ptr<const pg::PathGuidingSample> vEnd = vBegin + M;
        const device_ptr<int4> voxBegin(outVoxelGrids);
        thrust::transform(cuda::par.on(stream), vBegin, vEnd, voxBegin, SampleToVoxel{});
        thrust::sort(cuda::par.on(stream), voxBegin, voxBegin + M, Int4Less{});
        const auto uniqEnd = thrust::unique(cuda::par.on(stream), voxBegin, voxBegin + M, Int4Equal{});
        return static_cast<size_t>(uniqEnd - voxBegin); // K
    }


    struct GetX
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE int operator()(const int4& v) const
        {
            return v.x;
        }
    };
    struct GetY
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE int operator()(const int4& v) const
        {
            return v.y;
        }
    };
    struct GetZ
    {
        [[nodiscard]] FUNCTION_MODIFIER_INLINE int operator()(const int4& v) const
        {
            return v.z;
        }
    };
    inline std::tuple<int3, int3> GetMinMax(int4* voxelGrids, size_t count, cudaStream_t stream)
    {
        using namespace thrust;
        if (voxelGrids == nullptr || count == 0)
        {
            return { int3{0,0,0}, int3{0,0,0} };
        }
        const device_ptr<int4> begin(voxelGrids);
        const device_ptr<int4> end = begin + count;
        const auto xFirst = make_transform_iterator(begin, GetX{});
        const auto xLast  = make_transform_iterator(end,   GetX{});
        const auto mmX = minmax_element(thrust::cuda::par.on(stream), xFirst, xLast);
        const int minX = *mmX.first;
        const int maxX = *mmX.second;

        const auto yFirst = make_transform_iterator(begin, GetY{});
        const auto yLast  = make_transform_iterator(end,   GetY{});
        const auto mmY = minmax_element(thrust::cuda::par.on(stream), yFirst, yLast);
        const int minY = *mmY.first;
        const int maxY = *mmY.second;

        const auto zFirst = make_transform_iterator(begin, GetZ{});
        const auto zLast  = make_transform_iterator(end,   GetZ{});
        const auto mmZ = minmax_element(thrust::cuda::par.on(stream), zFirst, zLast);
        const int minZ = *mmZ.first;
        const int maxZ = *mmZ.second;
        return { int3{minX, minY, minZ}, int3{maxX, maxY, maxZ} };
    }

    FUNCTION_MODIFIER_INLINE size_t Flatten3D(const int x, const int y, const int z, const int3 dim)
    {
        return static_cast<size_t>(x) + static_cast<size_t>(dim.x) * (static_cast<size_t>(y) + static_cast<size_t>(dim.y) * static_cast<size_t>(z));
    }

    FUNCTION_MODIFIER_INLINE int3 DimsFromBoundsInclusive(const int3 mn, const int3 mx)
    {
        return make_int3(mx.x - mn.x + 1, mx.y - mn.y + 1, mx.z - mn.z + 1);
    }
    
    static __global__ void CopyOldIntoNewKernel(const size_t* __restrict__ oldVolume,
                                         int3 oldDim,
                                         size_t* __restrict__ newVolume,
                                         int3 newDim,
                                         int3 offset) // = oldMin - desiredMin
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= oldDim.x || y >= oldDim.y || z >= oldDim.z)
        {
            return;
        }
        int3 sourceLocation = make_int3(x, y, z);
        int3 targetLocation = make_int3(sourceLocation.x + offset.x, sourceLocation.y + offset.y, sourceLocation.z + offset.z);

        if (targetLocation.x < 0 || targetLocation.y < 0 || targetLocation.z < 0 || targetLocation.x >= newDim.x || targetLocation.y >= newDim.y || targetLocation.z >= newDim.z)
        {
            return;
        }
        size_t src = Flatten3D(sourceLocation.x, sourceLocation.y, sourceLocation.z, oldDim);
        size_t dst = Flatten3D(targetLocation.x, targetLocation.y, targetLocation.z, newDim);
        
        size_t srcBase = src * PATH_GUIDING_FACE_COUNT;
        size_t dstBase = dst * PATH_GUIDING_FACE_COUNT;
#pragma unroll
        for (int i=0; i<PATH_GUIDING_FACE_COUNT; ++i)
        {
            newVolume[dstBase + i] = oldVolume[srcBase + i];
        }
    }
    
    static inline cudaError_t CopyOldIntoNew(const size_t* oldVolume,
                               int3 boundMin, int3 boundMax,
                               size_t* newVolume,
                               int3 desiredBoundMin, int3 desiredBoundMax,
                               size_t defaultValue,
                               cudaStream_t stream)
    {
        using namespace thrust;
        if (!oldVolume || !newVolume)
        {
            return cudaErrorInvalidDevicePointer;
        }
        const int3 oldDim = DimsFromBoundsInclusive(boundMin, boundMax);
        const int3 newDim = DimsFromBoundsInclusive(desiredBoundMin, desiredBoundMax);

        if (oldDim.x <= 0 || oldDim.y <= 0 || oldDim.z <= 0 ||
            newDim.x <= 0 || newDim.y <= 0 || newDim.z <= 0)
        {
            return cudaErrorInvalidValue;
        }

        const size_t newCount = static_cast<size_t>(newDim.x) * newDim.y * newDim.z * PATH_GUIDING_FACE_COUNT;
        const device_ptr<size_t> newVolumePtr(newVolume);
        thrust::fill_n(thrust::cuda::par.on(stream), newVolumePtr, newCount, defaultValue);

        const int3 offset = make_int3(boundMin.x - desiredBoundMin.x,
                                      boundMin.y - desiredBoundMin.y,
                                      boundMin.z - desiredBoundMin.z);

        dim3 block(8, 8, 8);
        dim3 grid((oldDim.x + block.x - 1) / block.x,
                  (oldDim.y + block.y - 1) / block.y,
                  (oldDim.z + block.z - 1) / block.z);

        CopyOldIntoNewKernel<<<grid, block, 0, stream>>>(oldVolume, oldDim, newVolume, newDim, offset);
        return cudaGetLastError();
    }

    struct IsUnallocatedGrid
    {
        const size_t* volume; // int* [X*Y*Z*FACE]
        int3 start; // probeIndirectIndexVolumeStart_
        int3 dim; // probeIndirectIndexVolumeSize_ (= DimsFromBoundsInclusive(start,end))

        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(const int4& v) const
        {
            const int x = v.x - start.x;
            const int y = v.y - start.y;
            const int z = v.z - start.z;
            const int f = v.w;
            if (f < 0 || f >= PATH_GUIDING_FACE_COUNT)
            {
                return true;
            }
            if (x < 0 || y < 0 || z < 0 || x >= dim.x || y >= dim.y || z >= dim.z)
            {
                return true;
            }
            const size_t lin = static_cast<size_t>(x) + static_cast<size_t>(dim.x) * (static_cast<size_t>(y) + static_cast<size_t>(dim.y) * static_cast<size_t>(z));
            const size_t val = volume[lin * PATH_GUIDING_FACE_COUNT + f];
            return (val == SIZE_T_MAX);
        }
    };
    
    [[nodiscard]] inline size_t FilterUnallocatedGrids(const int4* uniqueGrids, size_t uniqueGridsCount,
                                                       const size_t* indirectVolume, int3 volStart, int3 volDim,
                                                       int4* outGrids,
                                                       cudaStream_t stream)
    {
        using namespace thrust;
        if (!uniqueGrids || uniqueGridsCount == 0 || !outGrids)
        {
            return 0;
        }
        device_ptr<const int4> inBegin(uniqueGrids);
        device_ptr<const int4> inEnd = inBegin + uniqueGridsCount;
        const device_ptr<int4> outBegin(outGrids);
        const IsUnallocatedGrid predicate{ indirectVolume, volStart, volDim };
        const auto outEnd = thrust::copy_if(thrust::cuda::par.on(stream),
                                            inBegin, inEnd,
                                            outBegin,
                                            predicate);
        return static_cast<size_t>(outEnd - outBegin);
    }

    static __global__ void ScatterAssignIndiceskernel(const int4* __restrict__ uniqueInvalidGrids, size_t uniqueInvalidGridsCount,
                                         size_t* __restrict__ indirectVolume,
                                         int3 volumeStart, int3 volumeDim,
                                         size_t baseIndex)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= uniqueInvalidGridsCount)
        {
            return;
        }
        const int4 v = uniqueInvalidGrids[i];
        const int lx = v.x - volumeStart.x;
        const int ly = v.y - volumeStart.y;
        const int lz = v.z - volumeStart.z;
        const int facetIndex  = v.w;

        if (facetIndex < 0 || facetIndex >= PATH_GUIDING_FACE_COUNT)
        {
            return;
        }
        if (lx < 0 || ly < 0 || lz < 0 || lx >= volumeDim.x || ly >= volumeDim.y || lz >= volumeDim.z)
        {
            return;
        }
        const size_t base = Flatten3D(lx,ly,lz,volumeDim) * PATH_GUIDING_FACE_COUNT;
        indirectVolume[base + facetIndex] = baseIndex + i;
    }

    static inline void ScatterAssignIndices(const int4* __restrict__ uniqueInvalidGrids, size_t uniqueInvalidGridsCount,
                                         size_t* __restrict__ indirectVolume,
                                         int3 volumeStart, int3 volumeDim,
                                         size_t baseIndex,
                                         cudaStream_t stream)
    {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((uniqueInvalidGridsCount + block.x - 1) / block.x));
        pg::ScatterAssignIndiceskernel<<<grid, block, 0, stream>>>(
            uniqueInvalidGrids, uniqueInvalidGridsCount,
            indirectVolume,
            volumeStart, volumeDim,
            baseIndex);
    }
    // for octahedron
    [[nodiscard]] FUNCTION_MODIFIER_DEVICE_INLINE int2 DirectionToLeafPixel(const float3 dir)
    {
        constexpr int res = Octahedron::resolution_;
        const float2 uv = SphereToUV(normalize(dir));
        int ix = floorToInt(uv.x * res);
        int iy = floorToInt(uv.y * res);
        ix = max(0, min(res - 1, ix));
        iy = max(0, min(res - 1, iy));
        return make_int2(ix, iy);
    }
    struct ProbeIndexLookup
    {
        const size_t* volume_;
        int3 volumeStart;
        int3 volumeSize;
        [[nodiscard]] FUNCTION_MODIFIER_INLINE size_t operator()(const int4 v) const
        {
            const int lx = v.x - volumeStart.x;
            const int ly = v.y - volumeStart.y;
            const int lz = v.z - volumeStart.z;
            const int f  = v.w;
            if (f < 0 || f >= PATH_GUIDING_FACE_COUNT)
            {
                return SIZE_T_MAX;
            }
            if (lx < 0 || ly < 0 || lz < 0 || lx >= volumeSize.x || ly >= volumeSize.y || lz >= volumeSize.z)
            {
                return SIZE_T_MAX;
            }
            const size_t lin = static_cast<size_t>(lx)
                             + static_cast<size_t>(volumeSize.x) * ( static_cast<size_t>(ly)
                             + static_cast<size_t>(volumeSize.y) * static_cast<size_t>(lz) );
            return volume_[lin * PATH_GUIDING_FACE_COUNT + static_cast<size_t>(f)];
        }
    };

    // lookup dont pass reference because need to copy from host to device
    static __global__ void AccumulateSamplesToOctaLeafKernel(const pg::PathGuidingSample* __restrict__ samples, size_t sampleNum, const ProbeIndexLookup lookup, Octahedron* __restrict__ tempProbes)
    {
        constexpr int res = Octahedron::resolution_;
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sampleNum; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        {
            const pg::PathGuidingSample currentSample = samples[i];
            if (!currentSample.valid_)
            {
                continue;
            }
            const size_t realIndex = lookup(currentSample.voxelGrid_);
            if (realIndex == SIZE_T_MAX)
            {
                continue;
            }
            float3 dir = normalize(currentSample.radianceDirection_);
            const int2 pix = DirectionToLeafPixel(dir);
            Octahedron& currentProbe = tempProbes[realIndex];
            
#ifdef USE_3_CHANNEL_PROBE
            float4* leaf = &currentProbe.data_[0];
#else
            float* leaf = &currentProbe.data_[0];
#endif
            const int idx = pix.y * res + pix.x;
            // J = dw/duv = 4*(|dx|+|dy|+|dz|)^3
            //const float sabs = fabsf(dir.x) + fabsf(dir.y) + fabsf(dir.z);
            //const float J = 4.0f * sabs * sabs * sabs;
            const float J = JacobianUVToOctahedronSphereDirection(dir);
#ifdef USE_3_CHANNEL_PROBE
            const float3 radianceMulJacobian = currentSample.radiance_ * J;
            atomicAdd(&leaf[idx].x, radianceMulJacobian.x);
            atomicAdd(&leaf[idx].y, radianceMulJacobian.y);
            atomicAdd(&leaf[idx].z, radianceMulJacobian.z);
#else
            const float radianceMulJacobian = currentSample.radianceStrength_ * J;
            atomicAdd(&leaf[idx], radianceMulJacobian);
#endif
            atomicAdd(&currentProbe.count_, 1ULL);
        }
    }

    inline void AccumulateSamplesToOctaLeaf(const pg::PathGuidingSample* validSamples, size_t validCount,
                                            const size_t* indirectVolume, int3 volStart, int3 volDim,
                                            Octahedron* probeDeviceTemp,
                                            cudaStream_t stream)
    {
        if (!validSamples || validCount == 0 || !probeDeviceTemp || !indirectVolume)
        {
            return;
        }
        const ProbeIndexLookup lookup{ indirectVolume, volStart, volDim };
        dim3 block(256);
        dim3 grid( static_cast<unsigned>((validCount + block.x - 1) / block.x) );
        AccumulateSamplesToOctaLeafKernel<<<grid, block, 0, stream>>>(validSamples, validCount, lookup, probeDeviceTemp);
    }
    // merge probe & temp
    static __global__ void MergeProbeKernel(Octahedron* __restrict__ probesOld, // non-temp real probe
                                            const Octahedron* __restrict__ probesNew, // temp
                                            size_t probeCount)
    {
        const size_t currentProbeIndex = blockIdx.x;
        if (currentProbeIndex >= probeCount)
        {
            return;
        }
        const Octahedron& probeNew = probesNew[currentProbeIndex];
        const unsigned long long countNew = probeNew.count_;
        if (countNew == 0ull)
        {
            return;
        }
        Octahedron& probeOld = probesOld[currentProbeIndex];
        const unsigned long long countOld = probeOld.count_;
        const double invDen = 1.0 / static_cast<double>(countOld + countNew);
        const double alpha = static_cast<double>(countOld) * invDen;

        constexpr int N = Octahedron::resolution2_;
        const int lane = threadIdx.x & 31; // mod 32
        
#ifdef USE_3_CHANNEL_PROBE
        for (int k = lane; k < N; k += 32)
        {
            float4 oldValue = probeOld.data_[k];
            const float4 newValue = probeNew.data_[k];
            oldValue.x = static_cast<float>(static_cast<double>(oldValue.x) * alpha + static_cast<double>(newValue.x) * invDen);
            oldValue.y = static_cast<float>(static_cast<double>(oldValue.y) * alpha + static_cast<double>(newValue.y) * invDen);
            oldValue.z = static_cast<float>(static_cast<double>(oldValue.z) * alpha + static_cast<double>(newValue.z) * invDen);
            probeOld.data_[k] = oldValue;
        }
#else
        float4* __restrict__ oldData4 = reinterpret_cast<float4*>(&probeOld.data_[0]);
        const float4* __restrict__ newData4 = reinterpret_cast<const float4*>(&probeNew.data_[0]);
        constexpr int V = N / 4;
        for (int j = lane; j < V; j += 32)
        {
            const float4 oldData = oldData4[j]; // old mean
            const float4 newData = newData4[j]; // new sum (this frame)

            float4 out;
            out.x = static_cast<float>(static_cast<double>(oldData.x) * alpha + static_cast<double>(newData.x) * invDen);
            out.y = static_cast<float>(static_cast<double>(oldData.y) * alpha + static_cast<double>(newData.y) * invDen);
            out.z = static_cast<float>(static_cast<double>(oldData.z) * alpha + static_cast<double>(newData.z) * invDen);
            out.w = static_cast<float>(static_cast<double>(oldData.w) * alpha + static_cast<double>(newData.w) * invDen);
            oldData4[j] = out;
        }
        if (((N & 3) != 0) && lane == 0)
        {
            for (int k = V * 4; k < N; ++k)
            {
                float* d = &probeOld.data_[0];
                const float* t = &probeNew.data_[0];
                d[k] = static_cast<float>(static_cast<double>(d[k]) * static_cast<double>(countOld) * invDen + static_cast<double>(t[k]) * invDen);
            }
        }
#endif
        if (lane == 0)
        {
            probeOld.count_ = min(PATH_GUIDING_PROBE_MAX_STORE_COUNT, countOld + countNew);
        }
        __syncthreads();
        if (lane == 0)
        {
            //__threadfence_block();
            probeOld.GenerateMipmaps();
#ifdef NDEBUG
#else
            //probeOld.DebugAssertMipmapAverage(); // always pass so safe
#endif
        }
    }

    inline void LaunchMergeProbeKernel(Octahedron* probe,
                                       const Octahedron* probeTemp,
                                       size_t probeCount,
                                       cudaStream_t stream)
    {
        if (!probe || !probeTemp || probeCount == 0)
        {
            return;
        }
        constexpr int threads = 32; // 1 warp
        const int blocks = static_cast<int>(probeCount);
        MergeProbeKernel<<<blocks, threads, 0, stream>>>(probe, probeTemp, probeCount);
    }
}