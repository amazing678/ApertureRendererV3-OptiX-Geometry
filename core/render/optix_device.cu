#include <optix_device.h>
#include "optix_params.h"

extern "C" { __constant__ OptixParams params; }

#define PI 3.14159265359f

// ==========================================
// sample.cuh 里的 GGX 算法
// ==========================================
__device__ float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float denom = (NdotH * NdotH * (a2 - 1.0f) + 1.0f);
    return a2 / max(PI * denom * denom, 1e-7f);
}

__device__ float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0f), roughness) * GeometrySchlickGGX(max(dot(N, L), 0.0f), roughness);
}

__device__ float3 fresnelSchlick(float cosTheta, float3 F0) {
    float p = powf(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
    return make_float3(F0.x + (1.0f - F0.x) * p, F0.y + (1.0f - F0.y) * p, F0.z + (1.0f - F0.z) * p);
}

// 1. Raygen
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float u = (float)idx.x / (float)dim.x;
    float v = (float)idx.y / (float)dim.y;
    float2 d = make_float2(u * 2.0f - 1.0f, (1.0f - v) * 2.0f - 1.0f);

    float3 origin = params.cam_eye;
    float3 direction = normalize(d.x * params.cam_u + d.y * params.cam_v + params.cam_w);

    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    optixTrace(params.handle, origin, direction, 0.0f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2, p3);

    if (p3 == 1) {
        params.image[idx.y * params.width + idx.x] = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    }
}

// 2. Miss (未命中时触发)
extern "C" __global__ void __miss__ms() {
    //阴影射线的核心：如果探测光线没有撞到任何阻挡物，就会执行 Miss，并把 p0 设为 1 (代表可见/被光照亮)
    optixSetPayload_0(1);
}

// 3. ClosestHit (物理光照 + 阴影生成)
extern "C" __global__ void __closesthit__ch() {
    float2 bary = optixGetTriangleBarycentrics();
    float u = bary.x; float v = bary.y; float w = 1.0f - u - v;

    int primIdx = optixGetPrimitiveIndex();
    SceneObject obj = params.objects[params.primToObjIndex[primIdx]];

    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float t = optixGetRayTmax();

    // 利用你的 vector.cuh 重载运算符，代码更清爽
    float3 hitPos = rayOrigin + rayDir * t;

    float3 n0 = obj.additionalObjectInfo_.triangleInfo_.v0_.normal_;
    float3 n1 = obj.additionalObjectInfo_.triangleInfo_.v1_.normal_;
    float3 n2 = obj.additionalObjectInfo_.triangleInfo_.v2_.normal_;
    float3 N = normalize(n0 * w + n1 * u + n2 * v); // 原始几何法线

    // 提取 UV 坐标
    float2 uv0 = obj.additionalObjectInfo_.triangleInfo_.v0_.uv_;
    float2 uv1 = obj.additionalObjectInfo_.triangleInfo_.v1_.uv_;
    float2 uv2 = obj.additionalObjectInfo_.triangleInfo_.v2_.uv_;
    float2 texUV = make_float2(uv0.x * w + uv1.x * u + uv2.x * v, uv0.y * w + uv1.y * u + uv2.y * v);

    // 材质属性
    float3 albedo = obj.material_.texture_.baseTexture_.albedo_;
    float metallic = obj.material_.texture_.baseTexture_.metallic_;
    float roughness = obj.material_.texture_.baseTexture_.roughness_;

    cudaTextureObject_t albedoTex = obj.material_.texture_.baseTexture_.albedoTex_;
    cudaTextureObject_t metallicTex = obj.material_.texture_.baseTexture_.metallicTex_;
    cudaTextureObject_t roughnessTex = obj.material_.texture_.baseTexture_.roughnessTex_;
    cudaTextureObject_t normalTex = obj.material_.texture_.baseTexture_.normalTex_;

    // 采样基础贴图
    if (albedoTex != 0) {
        float4 texColor = tex2D<float4>(albedoTex, texUV.x, texUV.y);
        albedo = make_float3(texColor.x * texColor.x, texColor.y * texColor.y, texColor.z * texColor.z); // sRGB 2 Linear
    }
    if (metallicTex != 0) metallic = tex2D<float4>(metallicTex, texUV.x, texUV.y).x;
    if (roughnessTex != 0) roughness = tex2D<float4>(roughnessTex, texUV.x, texUV.y).x;
    roughness = clamp(roughness, 0.05f, 1.0f);

    // ========================================================
    // 细节优化 1：极其严谨的法线贴图 (Bump) 扰动
    // ========================================================
    if (normalTex != 0) {
        float4 nTex = tex2D<float4>(normalTex, texUV.x, texUV.y);

        // 【注意】如果你觉得凹凸效果反了(凸起变成了坑)，把 nTex.y 前面加上负号： -(nTex.y * 2.0f - 1.0f)
        float3 tangentNormal = make_float3(nTex.x * 2.0f - 1.0f, nTex.y * 2.0f - 1.0f, nTex.z * 2.0f - 1.0f);

        float3 p0 = obj.additionalObjectInfo_.triangleInfo_.v0_.position_;
        float3 p1 = obj.additionalObjectInfo_.triangleInfo_.v1_.position_;
        float3 p2 = obj.additionalObjectInfo_.triangleInfo_.v2_.position_;

        float3 edge1 = p1 - p0;
        float3 edge2 = p2 - p0;
        float2 deltaUV1 = make_float2(uv1.x - uv0.x, uv1.y - uv0.y);
        float2 deltaUV2 = make_float2(uv2.x - uv0.x, uv2.y - uv0.y);

        float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        float f = (abs(det) > 1e-6f) ? 1.0f / det : 1.0f;

        float3 T = make_float3(
            f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x),
            f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y),
            f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z)
        );

        T = obj.objectToWorld_ * T;
        T = normalize(T - N * dot(N, T)); // Gram-Schmidt 正交化，确保 T 和 N 绝对垂直
        float3 B = normalize(cross(N, T)); // 叉乘得到副切线

        // 用 TBN 矩阵将贴图法线转换到世界空间，覆盖掉原来的平滑法线 N
        N = normalize(T * tangentNormal.x + B * tangentNormal.y + N * tangentNormal.z);
    }

    if (params.bShowNormal) {
        float3 c = make_float3(N.x * 0.5f + 0.5f, N.y * 0.5f + 0.5f, N.z * 0.5f + 0.5f);
        optixSetPayload_0(__float_as_uint(c.x)); optixSetPayload_1(__float_as_uint(c.y));
        optixSetPayload_2(__float_as_uint(c.z)); optixSetPayload_3(1);
        return;
    }

    float3 directLighting = make_float3(0.0f, 0.0f, 0.0f);
    float3 V = normalize(params.cam_eye - hitPos);
    float NoV = max(dot(N, V), 0.0f);

    float3 F0 = make_float3(
        0.04f * (1.0f - metallic) + albedo.x * metallic,
        0.04f * (1.0f - metallic) + albedo.y * metallic,
        0.04f * (1.0f - metallic) + albedo.z * metallic
    );

    // ========================================================
    // 细节优化 2：柔和衰减的主光源
    // ========================================================
    for (int i = 0; i < params.lightCount; ++i) {
        SceneObject lightObj = params.objects[params.lights[i].objectIndex_];

        float3 L_unnorm = lightObj.center_ - hitPos;
        float distSq = dot(L_unnorm, L_unnorm);
        float3 L = normalize(L_unnorm);
        float NoL = max(dot(N, L), 0.0f);

        if (NoL > 0.0f) {
            float3 H = normalize(V + L);
            float3 emission = lightObj.material_.texture_.baseTexture_.emissive_;

            // 引入光源半径缓冲，防止模型顶部过曝泛白
            float lightRadiusSq = dot(lightObj.extent_, lightObj.extent_) * 2.0f;
            float attenuation = 1.0f / (distSq + lightRadiusSq + 0.01f);

            // 稍微压低一点强光，让纹理更清晰
            float3 radiance = make_float3(emission.x * attenuation * 0.5f, emission.y * attenuation * 0.5f, emission.z * attenuation * 0.5f);

            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

            float3 specular = make_float3(NDF * G * F.x, NDF * G * F.y, NDF * G * F.z);
            float denom = max(4.0f * NoV * NoL, 0.001f);
            specular = make_float3(specular.x / denom, specular.y / denom, specular.z / denom);

            float3 kD = make_float3((1.0f - F.x) * (1.0f - metallic), (1.0f - F.y) * (1.0f - metallic), (1.0f - F.z) * (1.0f - metallic));

            directLighting.x += (kD.x * albedo.x / PI + specular.x) * radiance.x * NoL;
            directLighting.y += (kD.y * albedo.y / PI + specular.y) * radiance.y * NoL;
            directLighting.z += (kD.z * albedo.z / PI + specular.z) * radiance.z * NoL;
        }
    }

    // ========================================================
    // 细节优化 3：定向环境光 (让暗部的 Bump 彻底爆表！)
    // ========================================================
    // 这里利用扰动后的法线 N 来采样一个渐变的环境色
    float upFactor = N.y * 0.5f + 0.5f;

    // 天顶色(微蓝发亮) 和 地面色(暖黑)，你可以自由微调这两个颜色
    float3 skyColor = make_float3(0.25f, 0.30f, 0.40f);
    float3 groundColor = make_float3(0.02f, 0.02f, 0.02f);

    // 环境漫反射
    float3 ambientDiffuse = make_float3(
        albedo.x * (groundColor.x * (1.0f - upFactor) + skyColor.x * upFactor),
        albedo.y * (groundColor.y * (1.0f - upFactor) + skyColor.y * upFactor),
        albedo.z * (groundColor.z * (1.0f - upFactor) + skyColor.z * upFactor)
    );

    // 环境镜面反射 (让暗部的金属和光泽部分也能反光)
    float3 R = reflect(-V, N); // 视线在法线上的反射向量
    float specUpFactor = R.y * 0.5f + 0.5f;
    float3 ambientSpec = make_float3(
        (groundColor.x * (1.0f - specUpFactor) + skyColor.x * specUpFactor) * F0.x * (1.0f - roughness),
        (groundColor.y * (1.0f - specUpFactor) + skyColor.y * specUpFactor) * F0.y * (1.0f - roughness),
        (groundColor.z * (1.0f - specUpFactor) + skyColor.z * specUpFactor) * F0.z * (1.0f - roughness)
    );

    float3 finalColor = make_float3(
        directLighting.x + ambientDiffuse.x * (1.0f - metallic) + ambientSpec.x,
        directLighting.y + ambientDiffuse.y * (1.0f - metallic) + ambientSpec.y,
        directLighting.z + ambientDiffuse.z * (1.0f - metallic) + ambientSpec.z
    );

    optixSetPayload_0(__float_as_uint(finalColor.x));
    optixSetPayload_1(__float_as_uint(finalColor.y));
    optixSetPayload_2(__float_as_uint(finalColor.z));
    optixSetPayload_3(1);
}