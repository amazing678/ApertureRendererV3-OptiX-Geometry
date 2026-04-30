#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// 包含刚才下载的图片读取库
#include "stb_image.h"

inline cudaTextureObject_t LoadCUDATexture(const std::string& path)
{
    int width, height, channels;

    // 1. 强制加载为 4 通道 (RGBA)，因为 GPU 显存对 4 字节对齐的 float4/uchar4 支持最高效
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);

    if (!data)
    {
        printf("Failed to load texture: %s\n", path.c_str());
        return 0;
    }

    // 2. 描述通道格式 (8位无符号，RGBA 四个通道)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    // 3. 分配 CUDA 2D Array (显卡上专门用于纹理极速采样的硬件内存结构)
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 4. 将图片数据从 CPU 内存搬运到 GPU Array (注意 cudaMemcpy2DToArray 的宽度单位是 Byte)
    const size_t spitch = width * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, spitch, height, cudaMemcpyHostToDevice);

    // 5. 设置资源描述符，告诉 CUDA 数据在哪
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 6. 设置纹理采样描述符 (最核心的部分，决定了图片在 3D 模型上长什么样)
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;   // U 轴循环 (允许贴图平铺)
    texDesc.addressMode[1] = cudaAddressModeWrap;   // V 轴循环
    texDesc.filterMode = cudaFilterModeLinear;      // 线性插值 (放大不出现马赛克，实现平滑过渡)
    texDesc.readMode = cudaReadModeNormalizedFloat; // 硬件自动将 0~255 的像素值瞬间转换成 0.0~1.0 的浮点数
    texDesc.normalizedCoords = 1;                   // 告诉 GPU 我们的 UV 范围是 0.0~1.0，而不是像素绝对坐标

    // 7. 生成最终的无绑定纹理句柄 (Bindless Texture Handle)
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // 8. 数据已进显存，安全释放 CPU 上的原始图片数据
    stbi_image_free(data);
    printf("Texture loaded to GPU: %s (%dx%d)\n", path.c_str(), width, height);

    return texObj; // 返回这串类似长整型的句柄
}