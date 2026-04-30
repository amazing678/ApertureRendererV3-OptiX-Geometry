#pragma once
#include <optix.h>
#include "vector.cuh" //数学库 float3 等
#include "render/object.cuh"

struct OptixParams
{
    OptixTraversableHandle handle;

    float3* image;                 // 渲染输出的画面缓冲
    unsigned int width;
    unsigned int height;

    float3 cam_eye;                // 摄像机位置
    float3 cam_u, cam_v, cam_w;    // 摄像机方向向量

    SceneObject* objects;  // 物体数组
    int* primToObjIndex;   // 映射表

    bool bShowNormal; // GUI 开关：是否只显示法线
    AdditionalLightInfo* lights;
    int lightCount;
};