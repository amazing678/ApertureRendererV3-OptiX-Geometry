#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>

class MyOptixDenoiser {
private:
    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;

    CUdeviceptr stateBuffer = 0;
    size_t stateBufferSize = 0;

    CUdeviceptr scratchBuffer = 0;
    size_t scratchBufferSize = 0;

    CUdeviceptr intensityBuffer = 0;

    int currentWidth = 0;
    int currentHeight = 0;
    bool isInitialized = false;

public:
    MyOptixDenoiser();
    ~MyOptixDenoiser();

    // 执行降噪的核心接口
    void Invoke(cudaStream_t stream, const float4* d_color, const float4* d_albedo, const float4* d_normal, float4* d_output, int width, int height);

private:
    void Init(int w, int h);
    void Cleanup();
};