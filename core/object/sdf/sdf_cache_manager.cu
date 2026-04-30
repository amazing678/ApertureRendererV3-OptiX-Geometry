// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>

#include "sdf_cache_manager.cuh"
#include "sdf_master.cuh"
#include "sdf_volume.cuh"

SDFCacheManager::~SDFCacheManager()
{
    cudaStreamSynchronize(stream_);
    //
    sdf::SDFVolumeInfo* nullInfos = nullptr;
    uint32_t zero = 0;
    cudaMemcpyToSymbol(sdf::cachedSDFInfoPtrs, &nullInfos, sizeof(sdf::SDFVolumeInfo*));
    CHECK_ERROR();
    cudaMemcpyToSymbol(sdf::cachedSDFInfoPtrsCount, &zero, sizeof(uint32_t));
    CHECK_ERROR();
    //
    for (const sdf::SDFVolumeInfo& currentSDFVolumeInfo : hostInfos_)
    {
        if (currentSDFVolumeInfo.volumePtr_)
        {
            cudaFree(currentSDFVolumeInfo.volumePtr_);
        }
    }
    hostInfos_.clear();
    count_ = 0;

    if (deviceInfos_)
    {
        cudaFree(deviceInfos_);
        deviceInfos_ = nullptr;
    }
}

void SDFCacheManager::Clear()
{
    for (const auto& info : hostInfos_)
    {
        if (info.volumePtr_)
        {
            cudaFree(info.volumePtr_);
        }
    }
    hostInfos_.clear();
    count_ = 0;
    PublishToDevice();
}

uint32_t SDFCacheManager::Capacity() const
{
    return capacity_;
}

uint32_t SDFCacheManager::Count() const
{
    return count_;
}

void SDFCacheManager::PublishToDevice()
{
    cudaMemcpyToSymbolAsync(sdf::cachedSDFInfoPtrs, &deviceInfos_, sizeof(sdf::SDFVolumeInfo*), 0, cudaMemcpyHostToDevice, stream_);
    CHECK_ERROR();
    cudaMemcpyToSymbolAsync(sdf::cachedSDFInfoPtrsCount, &count_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream_);
    CHECK_ERROR();
}