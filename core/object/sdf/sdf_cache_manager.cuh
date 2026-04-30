// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"

#include <vector>
#include <stdint.h>
#include <stdexcept>

#include "sdf_master.cuh"
#include "sdf_volume.cuh"

namespace sdf
{
    template <class SDFFunctor>
    __global__ void FillSDFVolume(float* __restrict__ out,
                              const int3 dims,
                              [[maybe_unused]] const float3 pMin,
                              [[maybe_unused]] const float3 pMax,
                              SDFFunctor sdf,
                              SDFInfo sdfInfo)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= dims.x || y >= dims.y || z >= dims.z)
        {
            return;
        }
        const float nx = (dims.x > 1) ? static_cast<float>(x) / static_cast<float>(dims.x - 1) : 0.0f;
        const float ny = (dims.y > 1) ? static_cast<float>(y) / static_cast<float>(dims.y - 1) : 0.0f;
        const float nz = (dims.z > 1) ? static_cast<float>(z) / static_cast<float>(dims.z - 1) : 0.0f;
        const float3 pos01 = make_float3(nx, ny, nz);

        const size_t idx = static_cast<size_t>(z) * dims.x * dims.y
                         + static_cast<size_t>(y) * dims.x + x;
        out[idx] = sdf(pos01 - 0.5f, sdfInfo);
    }
} // namespace sdf

class SDFCacheManager
{
    private:
    sdf::SDFVolumeInfo* deviceInfos_ = nullptr; // device: float* fix size
    std::vector<sdf::SDFVolumeInfo> hostInfos_; // host ptrs
    uint32_t capacity_ = 0;
    uint32_t count_ = 0;
    cudaStream_t stream_ = nullptr;
    public:
    explicit SDFCacheManager(cudaStream_t stream, uint32_t capacity = 128) : capacity_(capacity ? capacity : 1), stream_(stream)
    {
        cudaMalloc(&deviceInfos_, capacity_ * sizeof(sdf::SDFVolumeInfo));
        CHECK_ERROR();
        PublishToDevice();
    }

    ~SDFCacheManager();

    SDFCacheManager(const SDFCacheManager&) = delete;
    SDFCacheManager& operator=(const SDFCacheManager&) = delete;
    SDFCacheManager(SDFCacheManager&&) = delete;
    SDFCacheManager& operator=(SDFCacheManager&&) = delete;

    // return a cache index
    template <class SDFFunctor>
    inline uint32_t AddVolume(int3 dims, float3 pMin, float3 pMax, const SDFFunctor& sdf, const SDFInfo& sdfInfo)

    {
        if (count_ >= capacity_)
        {
            throw std::runtime_error("SDFCacheManager capacity exceeded");
        }
        const size_t voxels = static_cast<size_t>(dims.x) * dims.y * dims.z;
        if (voxels == 0)
        {
            throw std::runtime_error("Invalid volume dims");
        }
        float* deviceVol = nullptr;
        cudaMalloc(&deviceVol, voxels * sizeof(float));
        CHECK_ERROR();

        dim3 block(8, 8, 8);
        dim3 grid((dims.x + block.x - 1) / block.x,
                  (dims.y + block.y - 1) / block.y,
                  (dims.z + block.z - 1) / block.z);
        sdf::FillSDFVolume<<<grid, block, 0, stream_>>>(deviceVol, dims, pMin, pMax, sdf, sdfInfo);
        CHECK_ERROR();
        cudaStreamSynchronize(stream_);
        CHECK_ERROR();
        //
        if (hostInfos_.empty())
        {
            hostInfos_.reserve(capacity_);
        }
        sdf::SDFVolumeInfo info{};
        info.volumePtr_ = deviceVol;
        info.dim_ = dims;
        info.ComputeVoxelSize();
        hostInfos_.push_back(info);
        count_++;
        //
        cudaMemcpyAsync(deviceInfos_, hostInfos_.data(), count_ * sizeof(sdf::SDFVolumeInfo), cudaMemcpyHostToDevice, stream_);
        CHECK_ERROR();
        cudaStreamSynchronize(stream_);
        CHECK_ERROR();
        PublishToDevice();
        return count_ - 1;
    }
    
    void Clear();
    uint32_t Capacity() const;
    uint32_t Count() const;
private:
    void PublishToDevice();
};