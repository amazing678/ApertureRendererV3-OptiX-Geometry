#include "optix_denoiser.cuh"
#include <iostream>

// 必须包含这个，告诉编译器动态链接显卡驱动中的 OptiX 神经网络
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

// 错误检查宏
#define CHECK_OPTIX(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX Error: " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

MyOptixDenoiser::MyOptixDenoiser() {}

MyOptixDenoiser::~MyOptixDenoiser() {
    Cleanup();
}

void MyOptixDenoiser::Cleanup() {
    if (denoiser) { optixDenoiserDestroy(denoiser); denoiser = nullptr; }
    if (context) { optixDeviceContextDestroy(context); context = nullptr; }
    if (stateBuffer) { cudaFree(reinterpret_cast<void*>(stateBuffer)); stateBuffer = 0; }
    if (scratchBuffer) { cudaFree(reinterpret_cast<void*>(scratchBuffer)); scratchBuffer = 0; }
    if (intensityBuffer) { cudaFree(reinterpret_cast<void*>(intensityBuffer)); intensityBuffer = 0; }
    isInitialized = false;
}

void MyOptixDenoiser::Init(int w, int h) {
    if (isInitialized && currentWidth == w && currentHeight == h) return;

    Cleanup();

    // 1. 初始化 OptiX 驱动
    CHECK_OPTIX(optixInit());

    // 2. 创建 OptiX 上下文
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;
    CHECK_OPTIX(optixDeviceContextCreate(0, &options, &context));

    // 3. 创建降噪器 (开启 HDR 模式，强制开启法线和颜色引导)
    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = 1;
    denoiserOptions.guideNormal = 1;
    CHECK_OPTIX(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser));

    // 4. 询问神经网络需要多大的显存来做张量计算
    OptixDenoiserSizes denoiserSizes;
    CHECK_OPTIX(optixDenoiserComputeMemoryResources(denoiser, w, h, &denoiserSizes));

    stateBufferSize = denoiserSizes.stateSizeInBytes;
    scratchBufferSize = denoiserSizes.withoutOverlapScratchSizeInBytes;

    cudaMalloc(reinterpret_cast<void**>(&stateBuffer), stateBufferSize);
    cudaMalloc(reinterpret_cast<void**>(&scratchBuffer), scratchBufferSize);
    cudaMalloc(reinterpret_cast<void**>(&intensityBuffer), sizeof(float));

    // 5. 将显存喂给降噪器完成 Setup
    CHECK_OPTIX(optixDenoiserSetup(
        denoiser, 0, w, h,
        stateBuffer, stateBufferSize,
        scratchBuffer, scratchBufferSize
    ));

    currentWidth = w;
    currentHeight = h;
    isInitialized = true;
    std::cout << "OptiX AI Denoiser Initialized for " << w << "x" << h << std::endl;
}

void MyOptixDenoiser::Invoke(cudaStream_t stream, const float4* d_color, const float4* d_albedo, const float4* d_normal, float4* d_output, int width, int height) {
    // 如果窗口大小改变，自动重新初始化
    if (!isInitialized || width != currentWidth || height != currentHeight) {
        Init(width, height);
    }

    // 组装 OptiX 图像结构体
    auto setupImage = [](const float4* ptr, int w, int h) -> OptixImage2D {
        OptixImage2D img = {};
        img.data = reinterpret_cast<CUdeviceptr>(ptr);
        img.width = w;
        img.height = h;
        img.rowStrideInBytes = w * sizeof(float4);
        img.pixelStrideInBytes = sizeof(float4);
        img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        return img;
        };

    OptixImage2D inputColor = setupImage(d_color, width, height);
    OptixImage2D inputAlbedo = setupImage(d_albedo, width, height);
    OptixImage2D inputNormal = setupImage(d_normal, width, height);
    OptixImage2D outputImage = setupImage(d_output, width, height);

    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = inputAlbedo;
    guideLayer.normal = inputNormal;

    OptixDenoiserLayer layer = {};
    layer.input = inputColor;
    layer.output = outputImage;

    // HDR 模型必须计算画面平均强度
    CHECK_OPTIX(optixDenoiserComputeIntensity(
        denoiser, stream, &inputColor, intensityBuffer, scratchBuffer, scratchBufferSize
    ));

    OptixDenoiserParams params = {};
    //params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    params.hdrIntensity = intensityBuffer;
    params.blendFactor = 0.0f; // 0.0 表示完全使用 AI 预测的结果，不要混合噪点图

    //发射！调用 Tensor Core 神经网络开始推理！
    CHECK_OPTIX(optixDenoiserInvoke(
        denoiser, stream, &params,
        stateBuffer, stateBufferSize,
        &guideLayer, &layer, 1,
        0, 0, // 屏幕偏移
        scratchBuffer, scratchBufferSize
    ));
}