// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "camera.hpp"
#include "helper/measure.cuh"

#include "platform.h"
#include <time.h>

int SaveBMP(unsigned char* image, int imageWidth, int imageHeight, const char* filename)
{
    unsigned char header[54] = {
        0x42, 0x4d, 0   , 0, 0, 0, 0   , 0,
        0   , 0   , 0x36, 0, 0, 0, 0x28, 0,
        0   , 0   , 0   , 0, 0, 0, 0   , 0,
        0   , 0   , 0x01, 0, 0x20, 0, 0, 0,
        0   , 0   , 0   , 0, 0x10, 0, 0, 0,
        0   , 0   , 0   , 0, 0   , 0, 0, 0,
        0   , 0   , 0   , 0, 0   , 0
    };
    for (int i = 0; i < imageWidth; i++)
    {
        for (int j = 0; j < imageHeight; j++)
        {
            int b = j + i * imageWidth;
            b *= 4;
            const int k = image[b];
            image[b] = image[b + 2];
            image[b + 2] = k;
        }
    }
    const long file_size = static_cast<long>(imageWidth) * static_cast<long>(imageHeight) * 4 + 54;
    header[2] = static_cast<unsigned char>(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    const long width = imageWidth;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    const long height = -imageHeight;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;
    
    char fileName[128];
    sprintf(fileName, "%s.bmp", filename);
    
    FILE* filePtr = fopen(fileName, "wb");
    if (!filePtr)
    {
        return -1;
    }
    fwrite(header, sizeof(unsigned char), 54, filePtr);
    fwrite(image, sizeof(unsigned char), static_cast<size_t>(imageWidth) * imageHeight * 4, filePtr);
    fclose(filePtr);
    return 0;
}

void Camera::RenderToFile(const std::string& path, const int sampleNum, const ToneMapFunction tonemapFunction, const float exp) const
{
    unsigned char* data = new unsigned char[resolutionX_ * resolutionY_ * 4];
    clock_t startTime = clock();
    const std::vector<float3> renderResult = sceneRenderer_->Render(int2{ resolutionX_, resolutionY_ }, GetPosition(), GetDirection(), sampleNum);
    printf("Rendering done in %.2f s.\n", static_cast<double>(clock() - startTime) / 1000.0f);
    for (int i = 0; i < resolutionY_; i++)
    {
        for (int j = 0; j < resolutionX_; j++)
        {
            const float3 value = renderResult[i * resolutionY_ + j] * exp;
#ifdef USE_SPECTRUM_RENDERING
            float3 res = tonemapFunction(XYZ2SRGBLinearD65(value));
#else
            float3 res = tonemapFunction(value);
#endif
            res = saturate(res);
            data[(i * resolutionY_ + j) * 4] = static_cast<unsigned char>(res.x * 255);
            data[(i * resolutionY_ + j) * 4 + 1] = static_cast<unsigned char>(res.y * 255);
            data[(i * resolutionY_ + j) * 4 + 2] = static_cast<unsigned char>(res.z * 255);
            data[(i * resolutionY_ + j) * 4 + 3] = static_cast<unsigned char>(255);
        }
    }
    SaveBMP(data, resolutionX_, resolutionY_, path.c_str());
    delete[] data;
}

void Camera::Render(float3* target, float3* oddTarget, float3* evenTarget, pg::PathGuidingSample* sampleTarget, denoise::ScreenGBuffer* screenGBuffer, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer, unsigned int* target2, const int2 size, const int frame, const int tone) const
{
    sceneRenderer_->Render(target, target2, size, GetPosition(), GetDirection(), frame, tone, oddTarget, evenTarget, sampleTarget, screenGBuffer, screenStatisticsBuffer);
    if (oddTarget && evenTarget && size.x * size.y > 0)
    {
        variance_ = DeviceMSELuminance(oddTarget, evenTarget, size.x * size.y) * 1000000.0;
    }
}
