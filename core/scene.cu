// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#include "scene.hpp"

#include <cassert>

#include "render/render.cuh"
#include "tonemap.cuh"
#include "config.cuh"

#include "platform.h"
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>

#include "denoise/denoise_statistic.cuh"
#include "spectrum/spectrum_lut.cuh"
#include "denoise/optix_denoiser.cuh"
#include "render/optix_params.h"

#include <cuda_fp16.h> //加入半精度支持

//#include <optix_function_table_definition.h>//optiX函数定义
#include <optix_stubs.h>
#include <iostream>


// 自定义 OptiX 日志回调函数
static void SceneOptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << std::endl;
}

__global__ void ConvertToOptiXFormat(
    const float3* __restrict__ inHDR,
    const denoise::ScreenGBuffer* __restrict__ inGBuffer,
    float4* __restrict__ outColor,
    float4* __restrict__ outAlbedo,
    float4* __restrict__ outNormal,
    int totalPixels)

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;

    // 1. 颜色 float3 -> float4
    float3 color = inHDR[idx];
    outColor[idx] = make_float4(color.x, color.y, color.z, 1.0f);

    // 如果 GBuffer 指针为空，或者像素点没打中，给默认 0
    if (inGBuffer) {
        const denoise::ScreenGBuffer& gb = inGBuffer[idx];
        // 2. 解压半精度反照率 (half3 -> float4)
        outAlbedo[idx] = make_float4(
            __half2float(gb.albedoX_),
            __half2float(gb.albedoY_),
            __half2float(gb.albedoZ_),
            1.0f);

        // 3. 解压半精度法线 (half3 -> float4)
        outNormal[idx] = make_float4(
            __half2float(gb.normalX_),
            __half2float(gb.normalY_),
            __half2float(gb.normalZ_),
            1.0f);
    }
    else {
        outAlbedo[idx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        outNormal[idx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
}

__global__ void ConvertFromOptiXFormat(
    const float4 * __restrict__ inColor,
    float3 * __restrict__ outHDR,
    int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;

    float4 c = inColor[idx];
    outHDR[idx] = make_float3(c.x, c.y, c.z);
}

__device__ float exposure = 1;

template<bool bDebugAlbedo, bool bDebugNormal, bool bDebugDepth, bool bDebugMetallic, bool bDebugRoughness,
        bool bDebugStatisticsNum, bool bDebugStatisticsMean, bool bDebugStatisticsM2, bool bDebugStatisticsM3>
__global__ void Tonemap(
    float3* __restrict__ target,
    const denoise::ScreenGBuffer* __restrict__ screenGBuffers,
    const denoise::ScreenStatisticsBuffer* __restrict__ screenStatisticsBuffers,
    unsigned int* __restrict__ target2, int2 size, int toneType)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size.x * size.y)
    {
        return;
    }
    if constexpr (bDebugAlbedo)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoX_)));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoY_)));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoZ_)));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugNormal)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalX_) * 0.5f + 0.5f));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalY_) * 0.5f + 0.5f));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalZ_) * 0.5f + 0.5f));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugDepth)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(1.0f / (currentGBuffer.depth_ + 1.0f));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugMetallic)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(static_cast<float>(currentGBuffer.metallic_));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugRoughness)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(static_cast<float>(currentGBuffer.roughness_));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsNum)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float heat = static_cast<float>(log(static_cast<double>(currentStatisticsBuffer.num_) + 1.0)) * 0.01f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat));
        target2[id] = 0xff000000 | (red << 16);
    }
    else if constexpr (bDebugStatisticsMean)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.mean_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsM2)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.M2DivNum_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsM3)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.M3DivNum_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else
    {
#ifdef USE_SPECTRUM_RENDERING
        const float3 result = XYZ2SRGBLinearD65(target[id]);
#else
        const float3 result = target[id];
#endif
        float3 value = result * exposure;
        if (toneType == 1)
        {
            value = Gamma(value);
        }
        else if (toneType == 2)
        {
            value = ACES(value);
        }
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(value.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(value.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(value.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
}

template<bool bUseBVH, bool bUseNEE, bool bUsePathGuiding>
__global__ void RenderCamera(float3* target, const int2 size, const float3 cameraOrigin, const float3 cameraDirection,
    float3* oddTarget = nullptr, float3* evenTarget = nullptr,
    pg::PathGuidingSample* radianceSampleBuffer = nullptr, denoise::ScreenGBuffer* screenGBuffer = nullptr, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer = nullptr) // radianceSampleBuffer must sized size.x * size.y * PATH_GUIDING_COLLECT_DEPTH
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // y
    if (i >= size.x || j >= size.y)
    {
        return;
    }
    const int idx = j * size.x + i;

    curandState seed;
    InitRand(&seed);
    float u = (i + Rand1(&seed)) / static_cast<float>(size.x);
    float v = (j + Rand1(&seed)) / static_cast<float>(size.y);

    float2 ndc = make_float2(u * 2.f - 1.f, (1.0 - v) * 2.f - 1.f);
    constexpr float FOV = 60.0f * 3.1415926f / 180.0f;
    const float aspect = static_cast<float>(size.x) / max(1.0f, static_cast<float>(size.y));
    const float tanHalfY  = tanf(0.5f * FOV);
    const float tanHalfX  = tanHalfY * aspect;

    const float sx = ndc.x * tanHalfX;
    const float sy = ndc.y * tanHalfY;

    const float3 right = normalize(cross(cameraDirection, float3{0.0, 1.0, 0.0}));
    const float3 up = normalize(cross(cameraDirection, right));
    const float3 rayDirection = normalize(cameraDirection + right * sx + up * sy);
    const float4 resultAndDistance = CalculateRadiance<bUseBVH, bUseNEE, bUsePathGuiding>(
        &seed, cameraOrigin, rayDirection,
        radianceSampleBuffer ? &radianceSampleBuffer[idx * PATH_GUIDING_COLLECT_DEPTH] : nullptr,
        screenGBuffer ? &screenGBuffer[idx] : nullptr,
        screenStatisticsBuffer ? &screenStatisticsBuffer[idx] : nullptr
    );

    float3 result = make_f3(resultAndDistance);
    result = max(float3{ 0 }, result);
    
    const float lerpRate = 1.0f / static_cast<float>(1 + FRAME_INDEX);
    target[idx] = lerp(target[idx], result, lerpRate);
    if(oddTarget != nullptr && evenTarget != nullptr)
    {
        const float lerpRateHalf = 1.0f / static_cast<float>(1 + (FRAME_INDEX >> 1));
        if(FRAME_INDEX % 2 == 1)
        {
            oddTarget[idx] = lerp(oddTarget[idx], result, lerpRateHalf);
        }
        else
        {
            evenTarget[idx] = lerp(evenTarget[idx], result, lerpRateHalf);
        }
    }
}

#define DISPATCH_RENDER_CAMERA(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer) do {\
        dim3 dimBlock(8, 4);\
        dim3 dimGrid;\
        dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;\
        dimGrid.y = (size.y + dimBlock.y - 1) / dimBlock.y;\
        cudaStreamWaitEvent(producerStream_, volumeWrittenEvent_, 0);\
        switch ((sceneSetting_.bUseBVH_?4:0) | (sceneSetting_.bUseNEE_?2:0) | (sceneSetting_.bUsePathGuiding_?1:0)) {\
            case 0: RenderCamera<false, false, false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 1: RenderCamera<false, false, true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 2: RenderCamera<false, true , false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 3: RenderCamera<false, true , true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 4: RenderCamera<true , false, false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 5: RenderCamera<true , false, true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 6: RenderCamera<true , true , false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 7: RenderCamera<true , true , true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
        }\
        cudaEventRecord(radianceSamplesReadyEvent_, producerStream_);\
    }while(0)

std::vector<float3> SceneRenderer::Render(const int2 size, const float3 cameraOrigin, const float3 cameraDirection, int sampleNum) 
{
    SetupSpectrumLUT();
    UpdateSceneToDeviceIfDirty();
    
    float3* target;
    cudaMalloc(&target, size.x * size.y * sizeof(float3));

    for (int i = 0; i < sampleNum; i++)
    {
        cudaMemcpyToSymbol(FRAME_INDEX, &i, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(TOTAL_FRAME_INDEX, &i, sizeof(int), 0, cudaMemcpyHostToDevice);

        DISPATCH_RENDER_CAMERA(target, size, cameraOrigin, cameraDirection, nullptr, nullptr, nullptr, nullptr, nullptr);

        cudaDeviceSynchronize();

        CHECK_ERROR();



    }

    std::vector<float3> resultCPU(size.x * size.y);

    cudaMemcpy(resultCPU.data(), target, sizeof(float3) * size.x * size.y, cudaMemcpyDeviceToHost);
    cudaFree(target);
    CHECK_ERROR();

    return resultCPU;
}

void SceneRenderer::Render(
    float3* hdrTarget, unsigned int* ldrTarget, const int2 size,
    const float3 cameraOrigin, const float3 cameraDirection, int frameIndex, int toneType,
    float3* oddTarget, float3* evenTarget, pg::PathGuidingSample* radianceSampleBuffer,
    denoise::ScreenGBuffer* screenGBuffer, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer) 
{
    SetupSpectrumLUT();
    UpdateSceneProbe();   
    UpdateSceneToDeviceIfDirty();

    // 【终极修复区】：如果外部没给 G-Buffer，我们自己造一个持久的！
    static denoise::ScreenGBuffer* internalGBuffer = nullptr;
    static denoise::ScreenStatisticsBuffer* internalStatBuffer = nullptr;
    static int internalBufferSize = 0;

    const int numel = size.x * size.y;
    if (internalBufferSize != numel) {
        if (internalGBuffer) cudaFree(internalGBuffer);
        if (internalStatBuffer) cudaFree(internalStatBuffer);
        cudaMalloc(&internalGBuffer, sizeof(denoise::ScreenGBuffer) * numel);
        cudaMalloc(&internalStatBuffer, sizeof(denoise::ScreenStatisticsBuffer) * numel);
        internalBufferSize = numel;
    }

    // 智能选择：外部有传就用外部的，没有就用我们内部的
    denoise::ScreenGBuffer* actualGBuffer = screenGBuffer ? screenGBuffer : internalGBuffer;
    denoise::ScreenStatisticsBuffer* actualStatBuffer = screenStatisticsBuffer ? screenStatisticsBuffer : internalStatBuffer;

    cudaMemcpyToSymbol(FRAME_INDEX, &frameIndex, sizeof(int), 0, cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpyToSymbol(TOTAL_FRAME_INDEX, &totalFrameIndex_, sizeof(int), 0, cudaMemcpyHostToDevice);    
    CHECK_ERROR();

    totalFrameIndex_++;
    //
    //DISPATCH_RENDER_CAMERA(hdrTarget, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer);
    DISPATCH_RENDER_CAMERA(hdrTarget, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, actualGBuffer, actualStatBuffer);
    CHECK_ERROR();

    if (optixPipeline_ && gasHandle_ && d_params_)
    {
        // 1. 在 CPU 计算摄像机矩阵，传给 GPU
        const float aspect = static_cast<float>(size.x) / max(1.0f, static_cast<float>(size.y));
        const float FOV = 60.0f * 3.1415926f / 180.0f;
        const float tanHalfY = tanf(0.5f * FOV);
        const float tanHalfX = tanHalfY * aspect;
        const float3 right = normalize(cross(cameraDirection, make_float3(0.0f, 1.0f, 0.0f)));
        const float3 up = normalize(cross(cameraDirection, right));

        OptixParams params;
        params.handle = gasHandle_;
        params.image = hdrTarget; //直接修改画面缓冲
        params.width = size.x;
        params.height = size.y;
        params.cam_eye = cameraOrigin;
        params.cam_u = right * tanHalfX;
        params.cam_v = up * tanHalfY;
        params.cam_w = cameraDirection;

        params.objects = sceneObjectsDevice_; // 你的主引擎本来就分配好的指针
        params.primToObjIndex = d_primToObjIndex_; // 我们刚建好的表

        //原有光源数据
        params.lights = sceneLightsInfoDevice_;
        params.lightCount = optixLightCount_;

        params.bShowNormal = sceneSetting_.bDebugNormal_;

        cudaMemcpy(d_params_, &params, sizeof(OptixParams), cudaMemcpyHostToDevice);

        // 2.硬件光线发射
        optixLaunch(optixPipeline_, producerStream_, (CUdeviceptr)d_params_, sizeof(OptixParams), &sbt_, size.x, size.y, 1);
        cudaStreamSynchronize(producerStream_);
    }


    if(sceneSetting_.bUsePathGuiding_)
    {
        UpdatePathGuidingSample();
        CHECK_ERROR();
    }
    else
    {
        const cudaStream_t stream = pathGuidingStream_;
        cudaStreamWaitEvent(pathGuidingStream_, radianceSamplesReadyEvent_, 0);
        CHECK_ERROR();
        // reset everything related to path guiding
        ResetPathGuidingSample(stream);
        CHECK_ERROR();
        cudaEventRecord(volumeWrittenEvent_, stream);
        CHECK_ERROR();
    }
    //
    int taskNum = size.x * size.y;
    int group = 32;
    int group_num = taskNum / group + (taskNum % group != 0 ? 1 : 0);
#define CALL_OUTPUT_FUNCTION(hdrTarget) <<<group_num, group>>>(hdrTarget, screenGBuffer, screenStatisticsBuffer, ldrTarget, size, toneType);
    if(sceneSetting_.bDebugAlbedo_)
    {
        Tonemap<true, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugNormal_)
    {
        Tonemap<false, true, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugDepth_)
    {
        Tonemap<false, false, true, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugMetallic_)
    {
        Tonemap<false, false, false, true, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugRoughness_)
    {
        Tonemap<false, false, false, false, true, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsNum_)
    {
        Tonemap<false, false, false, false, false, true, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsMean_)
    {
        Tonemap<false, false, false, false, false, false, true, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsM2_)
    {
        Tonemap<false, false, false, false, false, false, false, true, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsM3_)
    {
        Tonemap<false, false, false, false, false, false, false, false, true>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDenoise_)
    {
        // 【核心修复】：强制默认流等待，直到光线追踪 Kernel 完全写完 G-Buffer！
        cudaStreamSynchronize(producerStream_);

        const int numel = size.x * size.y;
        if(denoisedHDRDevice_ == nullptr || denoisedHDRDeviceAllocatedSize_ != numel)
        {
            if(denoisedHDRDevice_)
            {
                cudaFree(denoisedHDRDevice_);
                CHECK_ERROR();
            }
            cudaMalloc(&denoisedHDRDevice_, sizeof(float3) * numel);
            CHECK_ERROR();
            denoisedHDRDeviceAllocatedSize_ = numel;
        }

        // 3. 分配 OptiX 需要的 4 通道 float4 张量缓冲池
        static float4* d_optixColor = nullptr;
        static float4* d_optixAlbedo = nullptr;
        static float4* d_optixNormal = nullptr;
        static float4* d_optixOutput = nullptr;
        static int optixBufferSize = 0;

        if (optixBufferSize != numel) {
            if (d_optixColor) cudaFree(d_optixColor);
            if (d_optixAlbedo) cudaFree(d_optixAlbedo);
            if (d_optixNormal) cudaFree(d_optixNormal);
            if (d_optixOutput) cudaFree(d_optixOutput);

            cudaMalloc(&d_optixColor, numel * sizeof(float4));
            cudaMalloc(&d_optixAlbedo, numel * sizeof(float4));
            cudaMalloc(&d_optixNormal, numel * sizeof(float4));
            cudaMalloc(&d_optixOutput, numel * sizeof(float4));
            optixBufferSize = numel;
            CHECK_ERROR();
        }

        int blockSize = 256;
        int gridSize = (numel + blockSize - 1) / blockSize;

        // Step A: 解压半精度并打包成 float4 张量
        ConvertToOptiXFormat << <gridSize, blockSize >> > (hdrTarget, screenGBuffer, d_optixColor, d_optixAlbedo, d_optixNormal, numel);
        CHECK_ERROR();

        // Step B: 调用封装好的 OptiX 引擎执行 AI 降噪
        static MyOptixDenoiser aiDenoiser;
        aiDenoiser.Invoke(producerStream_, d_optixColor, d_optixAlbedo, d_optixNormal, d_optixOutput, size.x, size.y);

        // Step C: 将 float4 张量重新转回渲染管线的 float3
        // 注意：在完整写好 OptiX 类之前，这里先用原图 (d_optixColor) 喂给输出，确保你能跑通不黑屏！
        ConvertFromOptiXFormat << <gridSize, blockSize >> > (d_optixOutput, denoisedHDRDevice_, numel);
        // 等我们加上 OptiX 类后，上面这行要换成：ConvertFromOptiXFormat<<<...>>>(d_optixOutput, denoisedHDRDevice_, numel);

        /*const dim3 blockBilateral(16, 16);
        const dim3 gridBilateral( (size.x + blockBilateral.x - 1) / blockBilateral.x,
                                  (size.y + blockBilateral.y - 1) / blockBilateral.y );*/
        /*denoise::Bilateral16x16<7><<<gridBilateral, blockBilateral>>>(
            hdrTarget,
            denoisedHDRDevice_,
            screenGBuffer,
            screenStatisticsBuffer,
            size
        );*/
        //denoise::Bilateral16x16<7> << <gridBilateral, blockBilateral >> > (
        //    hdrTarget,
        //    denoisedHDRDevice_,
        //    actualGBuffer,      // <--- 换成这个
        //    actualStatBuffer,   // <--- 换成这个
        //    size
        //    );
        CHECK_ERROR();
        Tonemap<false, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(denoisedHDRDevice_)
        CHECK_ERROR();
    }
    else
    {
        Tonemap<false, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
#undef CALL_OUTPUT_FUNCTION
    //
    CHECK_ERROR();

    cudaDeviceSynchronize();
    CHECK_ERROR();

    CHECK_ERROR();
    return;
}

void SceneRenderer::ResetPathGuidingSample(const cudaStream_t stream)
{
    if (sharedCollectedRadianceSampleDevice_ && maxRadianceSampleCount_ > 0)
    {
        cudaMemsetAsync(sharedCollectedRadianceSampleDevice_, 0, maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample), pathGuidingStream_);
        CHECK_ERROR();
    }
    FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeDevice_));
    CHECK_ERROR();

    validRadianceSampleCount_ = 0;
    validSampleCount_ = 0;
    uniqueVoxelCount_ = 0;
    invalidUniqueVoxelCount_ = 0;
    probeCount_ = 0;
    probeIndirectIndexVolumeStart_ = make_int3(0,0,0);
    probeIndirectIndexVolumeEnd_ = make_int3(0,0,0);
    probeIndirectIndexVolumeSize_ = make_int3(0,0,0);
    //
    Octahedron* nullProbe = nullptr;
    size_t* nullIndex = nullptr;
    size_t zeroSizeT = 0;
    int3 zero3 = make_int3(0,0,0);
    // Probes
    cudaMemcpyToSymbolAsync(SCENE_PROBES, &nullProbe, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(SCENE_PROBES_COUNTS, &zeroSizeT, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR();

    // Indirect index volume + bounds
    cudaMemcpyToSymbolAsync(INDIRECT_INDEX_VOLUME, &nullIndex, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_START, &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_END,   &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_SIZE,  &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR();
}

void SceneRenderer::UpdatePathGuidingSample()
{
    const cudaStream_t stream = pathGuidingStream_;
    cudaStreamWaitEvent(pathGuidingStream_, radianceSamplesReadyEvent_, 0);
    CHECK_CUDA_SYNC();
    if(bNeedResetGuiding_)
    {
        ResetPathGuidingSample(stream);
        bNeedResetGuiding_ = false;
    }
    CHECK_CUDA_SYNC();
    if(validRadianceSampleCount_ != maxRadianceSampleCount_ || validCollectedRadianceSampleDevice_ == nullptr)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
        CHECK_CUDA_SYNC();
        FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
        CHECK_CUDA_SYNC();
        // reallocate
        cudaMalloc(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_), maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample));
        validRadianceSampleCount_ = maxRadianceSampleCount_;
        cudaMalloc(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_), maxRadianceSampleCount_ * sizeof(int4));
    }
    validSampleCount_ = pg::FilterValidSamples(sharedCollectedRadianceSampleDevice_, maxRadianceSampleCount_, validCollectedRadianceSampleDevice_, stream);
    CHECK_ERROR();
    //printf("validSampleCount_ vs totalSampleCount: %zd, %zd\n", validSampleCount_, maxRadianceSampleCount_);
    uniqueVoxelCount_ = pg::UniqueVoxelGridsFromValid(validCollectedRadianceSampleDevice_, validSampleCount_, uniqueProbeVolumeDevice_, stream);
    CHECK_ERROR();
    if(uniqueVoxelCount_ <= 0)
    {
        return;
    }
    //printf("uniqueVoxelCount_ vs validSampleCount_: %zd, %zd\n", uniqueVoxelCount_, validSampleCount_);
    const auto [voxelMin, voxelMax] = pg::GetMinMax(uniqueProbeVolumeDevice_, uniqueVoxelCount_, stream);
    CHECK_ERROR();
    //printf("voxel min: %d, %d, %d; voxel max: %d, %d, %d\n", voxelMin.x, voxelMin.y, voxelMin.z, voxelMax.x, voxelMax.y, voxelMax.z);
    const int3 desiredVolumeStart = {
        min(probeIndirectIndexVolumeStart_.x, voxelMin.x),
        min(probeIndirectIndexVolumeStart_.y, voxelMin.y),
        min(probeIndirectIndexVolumeStart_.z, voxelMin.z)
    };
    const int3 desiredVolumeEnd = {
        max(probeIndirectIndexVolumeEnd_.x, voxelMax.x),
        max(probeIndirectIndexVolumeEnd_.y, voxelMax.y),
        max(probeIndirectIndexVolumeEnd_.z, voxelMax.z)
    };
    if(probeIndirectIndexVolumeStart_.x != desiredVolumeStart.x || probeIndirectIndexVolumeStart_.y != desiredVolumeStart.y || probeIndirectIndexVolumeStart_.z != desiredVolumeStart.z ||
       probeIndirectIndexVolumeEnd_.x != desiredVolumeEnd.x || probeIndirectIndexVolumeEnd_.y != desiredVolumeEnd.y || probeIndirectIndexVolumeEnd_.z != desiredVolumeEnd.z)
    {
        const int3 newDim = pg::DimsFromBoundsInclusive(desiredVolumeStart, desiredVolumeEnd);
        CHECK_ERROR();
        const size_t voxels = static_cast<size_t>(newDim.x) * newDim.y * newDim.z;
        const size_t elements = voxels * PATH_GUIDING_FACE_COUNT;
        // need to reallocate
        if(probeIndirectIndexVolumeSize_.x == 0 || probeIndirectIndexVolumeSize_.y == 0 || probeIndirectIndexVolumeSize_.z == 0)
        {
            // allocate new memory
            cudaMalloc(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_), elements * sizeof(size_t));
            CHECK_ERROR();
            thrust::fill_n(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(probeIndirectIndexVolumeDevice_),
                           elements, SIZE_T_MAX);
            CHECK_ERROR();
            //printf("Allocated New Indirect Volume: %d, %d, %d.\n", newDim.x, newDim.y, newDim.z);
        }
        else
        {
            size_t* probeIndirectIndexVolumeDeviceOld = probeIndirectIndexVolumeDevice_;
            probeIndirectIndexVolumeDevice_ = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_), elements * sizeof(size_t));
            CHECK_ERROR();
            pg::CopyOldIntoNew(probeIndirectIndexVolumeDeviceOld,
                                      probeIndirectIndexVolumeStart_, probeIndirectIndexVolumeEnd_,
                                      probeIndirectIndexVolumeDevice_,
                                      desiredVolumeStart, desiredVolumeEnd,
                                      SIZE_T_MAX, stream); // default value
            CHECK_ERROR();
            FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDeviceOld));
            CHECK_ERROR();
            //printf("Expand Indirect Volume From %d, %d, %d To %d, %d, %d.\n", probeIndirectIndexVolumeSize_.x, probeIndirectIndexVolumeSize_.y, probeIndirectIndexVolumeSize_.z, newDim.x, newDim.y, newDim.z);
        }
        cudaMemcpyToSymbolAsync(INDIRECT_INDEX_VOLUME, &probeIndirectIndexVolumeDevice_, sizeof(size_t*), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        probeIndirectIndexVolumeStart_ = desiredVolumeStart;
        probeIndirectIndexVolumeEnd_ = desiredVolumeEnd;
        probeIndirectIndexVolumeSize_ = newDim;
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_START, &probeIndirectIndexVolumeStart_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_END, &probeIndirectIndexVolumeEnd_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_SIZE, &probeIndirectIndexVolumeSize_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        // update invalidUniqueVoxelDevice too in case not enough capacity
        FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
        cudaMalloc(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_), elements * sizeof(int4));
        CHECK_ERROR();
    }
    // filter invalid indirect voxel from uniqueProbeVolumeDevice_, uniqueVoxelCount_, probeIndirectIndexVolumeDevice_
    invalidUniqueVoxelCount_ = pg::FilterUnallocatedGrids(
                                uniqueProbeVolumeDevice_,
                                uniqueVoxelCount_,
                                probeIndirectIndexVolumeDevice_,
                                probeIndirectIndexVolumeStart_,
                                probeIndirectIndexVolumeSize_,
                                invalidUniqueVoxelDevice_,
                                stream);
    CHECK_ERROR();
    //printf("uniqueVoxelCount_ vs invalidUniqueVoxelCount_: %zd, %zd\n", uniqueVoxelCount_, invalidUniqueVoxelCount_);
    // alloc new probe
    if(invalidUniqueVoxelCount_ > 0)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
        Octahedron* probeDeviceOld = probeDevice_;
        const size_t newProbeCount = (probeCount_ + invalidUniqueVoxelCount_);
        cudaMalloc(reinterpret_cast<void**>(&probeDevice_), newProbeCount * sizeof(Octahedron));
        cudaMalloc(reinterpret_cast<void**>(&probeTempDevice_), newProbeCount * sizeof(Octahedron)); // temp
        cudaMemsetAsync(reinterpret_cast<char*>(probeDevice_) + probeCount_ * sizeof(Octahedron), 0, invalidUniqueVoxelCount_ * sizeof(Octahedron), stream);
        cudaMemsetAsync(reinterpret_cast<char*>(probeTempDevice_), 0, newProbeCount * sizeof(Octahedron), stream); // temp, all set to 0
        if (probeDeviceOld && probeCount_ > 0)
        {
            cudaMemcpy(probeDevice_, probeDeviceOld,
                       probeCount_ * sizeof(Octahedron),
                       cudaMemcpyDeviceToDevice);
            CHECK_ERROR();
        }
        CHECK_ERROR();
        FreeIfNotNullptr(reinterpret_cast<void**>(&probeDeviceOld));
        //
        pg::ScatterAssignIndices(invalidUniqueVoxelDevice_,
                                invalidUniqueVoxelCount_,
                                probeIndirectIndexVolumeDevice_,
                                probeIndirectIndexVolumeStart_,
                                probeIndirectIndexVolumeSize_,
                                probeCount_,
                                stream);
        probeCount_ = probeCount_ + invalidUniqueVoxelCount_;
        cudaMemcpyToSymbolAsync(SCENE_PROBES, &probeDevice_, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        cudaMemcpyToSymbolAsync(SCENE_PROBES_COUNTS, &probeCount_, sizeof(size_t), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        //printf("Write New Indirect Index Volume Count %zd: \n", invalidUniqueVoxelCount_);
        int4* invalidUniqueVoxelCPU = new int4[invalidUniqueVoxelCount_];
        cudaMemcpy(invalidUniqueVoxelCPU, invalidUniqueVoxelDevice_, sizeof(int4) * invalidUniqueVoxelCount_, cudaMemcpyDeviceToHost);
        CHECK_ERROR();
        //for(int i=0;i<invalidUniqueVoxelCount_;i++)
        {
            //printf("%d, %d, %d, %d\n", invalidUniqueVoxelCPU[i].x, invalidUniqueVoxelCPU[i].y, invalidUniqueVoxelCPU[i].z, invalidUniqueVoxelCPU[i].w);
        }
    }
    // write validCollectedRadianceSampleDevice_, validSampleCount_ to probeDeviceTemp_,
    if(validSampleCount_ > 0)
    {
        pg::AccumulateSamplesToOctaLeaf(validCollectedRadianceSampleDevice_, validSampleCount_,
                                                probeIndirectIndexVolumeDevice_, probeIndirectIndexVolumeStart_, probeIndirectIndexVolumeSize_,
                                                probeTempDevice_,
                                                stream);
        CHECK_ERROR();
        // merge all, TODO: merge those only updated
        pg::LaunchMergeProbeKernel(probeDevice_, probeTempDevice_, probeCount_, stream);
        CHECK_ERROR();
        cudaMemset(reinterpret_cast<char*>(probeTempDevice_), 0, probeCount_ * sizeof(Octahedron)); // temp, all set to 0
        CHECK_ERROR();
    }
    cudaEventRecord(volumeWrittenEvent_, stream);
}

void SceneRenderer::FreeSceneRelatedDevicePtr()
{
    FreeIfNotNullptr(reinterpret_cast<void**>(&bvhDevice_.objectIndices_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&bvhDevice_.nodes_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&sceneObjectsDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&sceneLightsInfoDevice_));
    // dont free screen size related device ptr like probeDevice_, collectedRadianceSampleDevice_

    //清理旧optiX显存
    if (d_vertices_) { cudaFree(d_vertices_); d_vertices_ = nullptr; }
    if (d_gasOutputBuffer_) { cudaFree(d_gasOutputBuffer_); d_gasOutputBuffer_ = nullptr; }
    if (d_primToObjIndex_) { cudaFree(d_primToObjIndex_); d_primToObjIndex_ = nullptr; }
    gasHandle_ = 0;
}

void SceneRenderer::FreeScreenRelatedDevicePtr()
{
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
}

void SceneRenderer::SetupSpectrumLUT()
{
#ifdef USE_SPECTRUM_RENDERING
    if(spectrumRGBLutDevice_ == nullptr || spectrumLambdaLutDevice_ == nullptr)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&spectrumRGBLutDevice_));
        FreeIfNotNullptr(reinterpret_cast<void**>(&spectrumLambdaLutDevice_));
        constexpr size_t rgbLutSize = SPECTRUM_RGB_LUT_RES * SPECTRUM_RGB_LUT_RES * SPECTRUM_RGB_LUT_RES * spectrum::query::KERNEL * sizeof(float);
        constexpr size_t lambdaLutSize = SPECTRUM_LAMBDA_LUT_RES * spectrum::query::KERNEL * sizeof(float);
        cudaMalloc(reinterpret_cast<void**>(&spectrumRGBLutDevice_), rgbLutSize);
        cudaMalloc(reinterpret_cast<void**>(&spectrumLambdaLutDevice_), lambdaLutSize);
        spectrum::PrecomputeSpectrumLUTsToDevice(spectrumRGBLutDevice_, spectrumLambdaLutDevice_);
        cudaMemcpyToSymbol(SPECTRUM_LUT_RGB,    &spectrumRGBLutDevice_,    sizeof(spectrumRGBLutDevice_));
        cudaMemcpyToSymbol(SPECTRUM_LUT_LAMBDA, &spectrumLambdaLutDevice_, sizeof(spectrumLambdaLutDevice_));
    }
#endif
}

void SceneRenderer::SetupStream()
{
    if(pathGuidingStream_ == nullptr)
    {
        cudaStreamCreateWithFlags(&producerStream_, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&pathGuidingStream_, cudaStreamNonBlocking);
        
        cudaEventCreateWithFlags(&radianceSamplesReadyEvent_,  cudaEventDisableTiming);
        cudaEventCreateWithFlags(&volumeWrittenEvent_, cudaEventDisableTiming);
        cudaEventRecord(volumeWrittenEvent_, pathGuidingStream_);
    }
}

void SceneRenderer::UpdateSceneProbe()
{
    //if(true)
    /*
    if(bSceneDirty_)
    {
        Octahedron testProbe;
        for(uint16_t x=0u;x< Octahedron::resolution_;x++)
        {
            for(uint16_t y=0u;y< Octahedron::resolution_;y++)
            {
#ifdef USE_3_CHANNEL_PROBE
                testProbe(x, y) = float3{
                    Hash11f(static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 16)) * 0.5f + 0.5f,
                    Hash11f((static_cast<uint32_t>(x) + 1314) | ((static_cast<uint32_t>(y) + 1314) << 16)) * 0.5f + 0.5f,
                    Hash11f((static_cast<uint32_t>(x) + 3756314) | ((static_cast<uint32_t>(y) + 3756314) << 16)) * 0.5f + 0.5f
                    };
#else
                testProbe(x, y) = Hash11f(static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 16)) * 0.5f + 0.5f;
#endif
            }
        }
        testProbe.GenerateMipmaps();
        cudaMalloc(reinterpret_cast<void**>(&probeDevice_), 1 * sizeof(Octahedron));
        cudaMemcpy(probeDevice_, &testProbe, 1 * sizeof(Octahedron), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_PROBES, &probeDevice_, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        constexpr int probeCount = 1;
        cudaMemcpyToSymbol(SCENE_PROBES_COUNTS, &probeCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
    }
    */
}

void SceneRenderer::BuildOptixGAS()
{
    if (!optixContext_) return;

    // 1. 收集世界坐标下的三角形顶点
    std::vector<float3> h_vertices;
    std::vector<int> h_primToObj; //记录索引的临时数组

    for (int i = 0; i < sceneObjects_.size(); i++)
    {
        auto& obj = sceneObjects_[i];
        // 提取所有的三角形
        if (obj.type_ == EObjectType::OBJ_TRIANGLE)
        {
            float3 v0 = obj.additionalObjectInfo_.triangleInfo_.v0_.position_;
            float3 v1 = obj.additionalObjectInfo_.triangleInfo_.v1_.position_;
            float3 v2 = obj.additionalObjectInfo_.triangleInfo_.v2_.position_;

            // 从局部坐标转换到真实的世界坐标
            v0 = obj.objectToWorld_ * v0;
            v1 = obj.objectToWorld_ * v1;
            v2 = obj.objectToWorld_ * v2;

            h_vertices.push_back(v0);
            h_vertices.push_back(v1);
            h_vertices.push_back(v2);

            h_primToObj.push_back(i);//三角形属于第i个sceneobject
        }
    }

    if (h_vertices.empty()) return; // 场景里没有模型，就不建树

    // 2. 将顶点全数推入显存
    const size_t verticesSize = h_vertices.size() * sizeof(float3);
    if (d_vertices_) cudaFree(d_vertices_);
    cudaMalloc(&d_vertices_, verticesSize);
    cudaMemcpy(d_vertices_, h_vertices.data(), verticesSize, cudaMemcpyHostToDevice);
    //映射表存入显存
    const size_t indexSize = h_primToObj.size() * sizeof(int);
    if (d_primToObjIndex_) cudaFree(d_primToObjIndex_);
    cudaMalloc(&d_primToObjIndex_, indexSize);
    cudaMemcpy(d_primToObjIndex_, h_primToObj.data(), indexSize, cudaMemcpyHostToDevice);

    // 3. 告诉 OptiX 数据的格式
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices_);
    triangleInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    triangleInput.triangleArray.numVertices = static_cast<unsigned int>(h_vertices.size());
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);

    // 禁用 AnyHit 着色器可以获得极致的硬件求交速度
    unsigned int triangleFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    triangleInput.triangleArray.flags = &triangleFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    // 4. 配置构建选项 (优先极速追踪)
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // 5. 让驱动计算需要多大的显存来建树
    OptixAccelBufferSizes gasBufferSizes;
    optixAccelComputeMemoryUsage(optixContext_, &accelOptions, &triangleInput, 1, &gasBufferSizes);

    // 6. 分配建树所需的临时和输出显存
    void* d_tempBuffer = nullptr;
    cudaMalloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes);
    cudaMalloc(&d_gasOutputBuffer_, gasBufferSizes.outputSizeInBytes);

    // 7.发号施令：命令 RT Cores 浇筑底层硬件树！
    OptixResult res = optixAccelBuild(
        optixContext_,
        producerStream_,
        &accelOptions,
        &triangleInput,
        1,
        reinterpret_cast<CUdeviceptr>(d_tempBuffer),
        gasBufferSizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(d_gasOutputBuffer_),
        gasBufferSizes.outputSizeInBytes,
        &gasHandle_,
        nullptr,
        0
    );

    cudaStreamSynchronize(producerStream_);
    cudaFree(d_tempBuffer); // 临时显存用完即扔

    if (res == OPTIX_SUCCESS) {
        std::cout << "[OptiX GAS] 硬件加速树构建成功！注入三角形数量: " << h_vertices.size() / 3 << std::endl;
    }
    else {
        std::cerr << "[OptiX GAS] 构建失败！错误码: " << res << std::endl;
    }
}

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<int> EmptyRecord;

void SceneRenderer::BuildOptixPipeline()
{
    if (!optixContext_) return;

    // 1. 读取我们刚编译出来的 PTX 文件
    std::ifstream file("D:/c++/ApertureRendererV3-OptiX-Geometry/build/ptx/optix_device.ptx", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "找不到 PTX 文件！请检查路径。" << std::endl;
        return;
    }
    std::string ptx((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // 2. 编译模块 (Module)
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 4;   // 我们要传回 R, G, B 三个值
    pipelineCompileOptions.numAttributeValues = 2; // 重心坐标 u, v
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params"; // 对应 optix_device.cu 里的名字

    optixModuleCreate(optixContext_, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), nullptr, nullptr, &optixModule_);

    // 3. 创建着色器组 (Raygen, Miss, ClosestHit)
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module = optixModule_;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    optixProgramGroupCreate(optixContext_, &rgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPG_);

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module = optixModule_;
    msDesc.miss.entryFunctionName = "__miss__ms";
    optixProgramGroupCreate(optixContext_, &msDesc, 1, &pgOptions, nullptr, nullptr, &missPG_);

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH = optixModule_;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    optixProgramGroupCreate(optixContext_, &hgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPG_);

    // 4. 将着色器组装成完整的管线 (Pipeline)
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;
    OptixProgramGroup programGroups[] = { raygenPG_, missPG_, hitgroupPG_ };
    optixPipelineCreate(optixContext_, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, nullptr, nullptr, &optixPipeline_);

    // 5. 构建 Shader Binding Table (SBT)
    EmptyRecord rgRecord, msRecord, hgRecord;
    optixSbtRecordPackHeader(raygenPG_, &rgRecord);
    optixSbtRecordPackHeader(missPG_, &msRecord);
    optixSbtRecordPackHeader(hitgroupPG_, &hgRecord);

    void* d_rg, * d_ms, * d_hg;
    cudaMalloc(&d_rg, sizeof(EmptyRecord));
    cudaMalloc(&d_ms, sizeof(EmptyRecord));
    cudaMalloc(&d_hg, sizeof(EmptyRecord));
    cudaMemcpy(d_rg, &rgRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms, &msRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hg, &hgRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    sbt_.raygenRecord = (CUdeviceptr)d_rg;
    sbt_.missRecordBase = (CUdeviceptr)d_ms;
    sbt_.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = (CUdeviceptr)d_hg;
    sbt_.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt_.hitgroupRecordCount = 1;

    // 分配并记录全局参数缓冲
    cudaMalloc(&d_params_, sizeof(OptixParams));
    std::cout << "[OptiX Pipeline] 渲染管线与 SBT 构建完毕，准备发射光线！" << std::endl;
}


void SceneRenderer::UpdateSceneToDeviceIfDirty()
{
    if(bSceneDirty_)
    {
        CHECK_ERROR();
        // update scene setting first
        cudaMemcpyToSymbol(SETTING, &sceneSetting_, sizeof(SceneSetting), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        // release old resource
        FreeSceneRelatedDevicePtr();
        CHECK_ERROR();

        BuildOptixGAS();//硬件bvh建树

        for (auto& obj : sceneObjects_) {
            if (obj.type_ == EObjectType::OBJ_TRIANGLE) {
                obj.type_ = (EObjectType)999;
            }
        }


        // update bvh
        std::vector<PrimitiveProxy> proxys;
        proxys.reserve(sceneObjects_.size());
        for (auto& sceneObject : sceneObjects_)
        {
            proxys.push_back(sceneObject.proxy_);
        }
        bvh_ = BuildBVH(proxys);
        // filter out lights
        std::vector<AdditionalLightInfo> lightsInfo;
        float totalLightWeightSum = 0.0f;
        for(int i=0;i<sceneObjects_.size();i++)
        {
            const auto& currentObject = sceneObjects_[i];
            sceneObjects_[i].objectIndex_ = i; // self map
            if(sceneObjects_[i].material_.shadingModel_ == EShadingModel::MAT_LIGHT)
            {
                sceneObjects_[i].lightIndex_ = static_cast<int>(lightsInfo.size()); // light map
                
                AdditionalLightInfo currentInfo;
                currentInfo.objectIndex_ = i;
                currentInfo.importanceWeight_ = fmaxf(currentObject.GetArea() * currentObject.GetEmissivePower(), 1.0e-20f);
                totalLightWeightSum += currentInfo.importanceWeight_;
                lightsInfo.push_back(currentInfo);
            }
            else
            {
                sceneObjects_[i].lightIndex_ = -1;
            }
        }
        for (auto& lightInfo : lightsInfo)
        {
            lightInfo.selectProb_ = lightInfo.importanceWeight_ / totalLightWeightSum;
        }
        cudaMalloc(reinterpret_cast<void**>(&sceneLightsInfoDevice_), lightsInfo.size() * sizeof(AdditionalLightInfo));
        cudaMemcpy(sceneLightsInfoDevice_, lightsInfo.data(), lightsInfo.size() * sizeof(AdditionalLightInfo), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_LIGHTS_INFO, &sceneLightsInfoDevice_, sizeof(AdditionalLightInfo*), 0, cudaMemcpyHostToDevice);
        const int lightCount = static_cast<int>(lightsInfo.size());
        cudaMemcpyToSymbol(SCENE_LIGHTS_COUNTS, &lightCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        optixLightCount_ = lightCount;//光源数量存入显存

        // copy bvh to device
        cudaMalloc(reinterpret_cast<void**>(&bvhDevice_.nodes_), bvh_.nodes_.size() * sizeof(BVHNode));
        cudaMemcpy(bvhDevice_.nodes_, bvh_.nodes_.data(), bvh_.nodes_.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(BVH_NODES, &bvhDevice_.nodes_, sizeof(BVHNode*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
	    const int nodeCount = static_cast<int>(bvh_.nodes_.size());
        cudaMemcpyToSymbol(BVH_NODES_COUNTS, &nodeCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        cudaMalloc(reinterpret_cast<void**>(&bvhDevice_.objectIndices_), bvh_.objectIndices_.size() * sizeof(int));
        cudaMemcpy(bvhDevice_.objectIndices_, bvh_.objectIndices_.data(), bvh_.objectIndices_.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(BVH_OBJECT_INDICES, &bvhDevice_.objectIndices_, sizeof(int*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        assert(bvh_.objectIndices_.size() == sceneObjects_.size());
        //
        cudaMalloc(reinterpret_cast<void**>(&sceneObjectsDevice_), sceneObjects_.size() * sizeof(SceneObject));
        cudaMemcpy(sceneObjectsDevice_, sceneObjects_.data(), sceneObjects_.size() * sizeof(SceneObject), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_OBJECTS, &sceneObjectsDevice_, sizeof(SceneObject*), 0, cudaMemcpyHostToDevice);
	    const int objectCount = static_cast<int>(sceneObjects_.size());
        cudaMemcpyToSymbol(SCENE_OBJECT_COUNTS, &objectCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        for (auto& obj : sceneObjects_) {
            if (obj.type_ == (EObjectType)999) {
                obj.type_ = EObjectType::OBJ_TRIANGLE; // 恢复三角形身份！
            }
        }

        bSceneDirty_ = false;
    }
}

SceneRenderer::SceneRenderer(const std::vector<SceneObject>& sceneObjects)
{
    SetupStream();

    //初始化optiX环境
    OptixResult res = optixInit();
    if (res != OPTIX_SUCCESS)
    {
        std::cerr << "optixInit 失败！错误码：" << res << std::endl;
    }
    else
    {
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &SceneOptixLogCallback;
        options.logCallbackLevel = 4; // 4 代表打印 INFO 级别的所有详细信息

        // 利用当前默认的 CUDA 上下文 (传 0 即可) 来创建 OptiX Context
        res = optixDeviceContextCreate(0, &options, &optixContext_);
        if (res != OPTIX_SUCCESS)
        {
            std::cerr << "OptiX Context 创建失败！" << std::endl;
        }
        else
        {
            std::cout << "[Hello OptiX] OptiX Context 初始化成功！RT Cores 待命！" << std::endl;
            BuildOptixPipeline(); //构建管线
        }
    }


    sdfManager_ = new SDFCacheManager(producerStream_);
    sceneObjects_ = sceneObjects; // copy
    CHECK_ERROR();
    MarkAsDirty();
    printf("Scene Loaded \n");
    printf("Memory allocated \n");
}

SceneRenderer::~SceneRenderer() 
{
    delete sdfManager_;
    FreeSceneRelatedDevicePtr();
    FreeScreenRelatedDevicePtr();
}

void SceneRenderer::SetExposure(float exp)
{
    cudaMemcpyToSymbol(exposure, &exp, sizeof(float), 0, cudaMemcpyHostToDevice);
}