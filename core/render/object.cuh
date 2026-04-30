// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "color.cuh"
#include "bvh/bvh.cuh"
#include "spectrum/spectrum.cuh"
#include "spectrum/spectrum_basis.cuh"

struct alignas(1) SceneSetting
{
    bool bUseBVH_ = true;
    bool bUseNEE_ = true;
    bool bUsePathGuiding_ = true;
    bool bDenoise_ = false;

    bool bDebugAlbedo_ = false;
    bool bDebugNormal_ = false;
    bool bDebugDepth_ = false;
    bool bDebugMetallic_ = false;
    bool bDebugRoughness_ = false;
    
    bool bDebugStatisticsNum_ = false;
    bool bDebugStatisticsMean_ = false;
    bool bDebugStatisticsM2_ = false;
    bool bDebugStatisticsM3_ = false;
};

enum class EObjectType
{
    OBJ_CUBE = 0,
    OBJ_SPHERE = 1,
    OBJ_SDF = 2,
    OBJ_TRIANGLE = 3,//三角形类型
};

enum class ESDFType
{
    SDF_TORUS = 0,
    SDF_VOLUME = 1,
    SDF_UNDEFINED = 2,
    SDF_MAX = 3, // or undefined
};

enum class EShadingModel
{
    MAT_DIFFUSE = 0,
    MAT_LIGHT = 1,
    MAT_GLASS = 2,
    MAT_GGX = 3,
    MAT_MAX = 4,
};

enum class ETextureType
{
    TEX_PURE_COLOR = 0,
};

// TODO: normal in gbuffer
struct __align__(16) GBuffer
{
    float3 albedo_ = float3{1.0f, 1.0f, 1.0f}; // linear space
    float3 emissive_ = float3{0.0f, 0.0f, 0.0f}; // linear space
    float specular_ = 0.0f;
    float metallic_ = 0.0f;
    float roughness_ = 0.0f;
    float IOR_ = 1.5f;
    float sigmaA_ = 100.0f;
    float3 normal_ = {0.0f, 0.0f, 1.0f};
    float abbeNumber_ = 20.0f; // 1.0 ~ 100.0
    float tiltNumber_ = 0.0f; // -1.0~1.0

    //4张贴图,0代表不使用贴图
    cudaTextureObject_t albedoTex_ = 0;
    cudaTextureObject_t normalTex_ = 0;
    cudaTextureObject_t metallicTex_ = 0;
    cudaTextureObject_t roughnessTex_ = 0;

#ifdef USE_SPECTRUM_RENDERING
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float GetIOR(const float lambda_nm) const
#else
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float GetIOR() const
#endif
    {
#ifdef USE_SPECTRUM_RENDERING
        return spectrum::IORFromAbbeNM(lambda_nm, IOR_, abbeNumber_);
#else
        return IOR_;
#endif
    }
};

__align__(16) struct IntersectionContext
{
    bool bHit_ = false;
    int objectIndex_ = -1;
    int lightIndex_ = -1; // only valid if hit TRUE LIGHT
    float distance_ = inf();

    float3 hitPosition_;
    float3 hitNormal_;
    
    float3 hitLocalPosition_;
    float3 hitLocalNormal_;
    GBuffer gbuffer_ = {};

    float2 uv_ = { 0.0f, 0.0f };//记录UV位置
    bool isInside_ = false;//区分光对于物体表面的位置是内还是外
};

struct OverlapContext
{
    bool bOverlap_ = false;
    int objectIndex_ = -1;
    
    float3 overlapPosition_;
};

struct Texture
{
    ETextureType textureType_ = ETextureType::TEX_PURE_COLOR;
    GBuffer baseTexture_ = {};
};
struct Material
{
    EShadingModel shadingModel_ = EShadingModel::MAT_DIFFUSE;
    Texture texture_ = {}; // TODO: texture / single point / procedural

    [[nodiscard]] FUNCTION_MODIFIER_STATIC Material CreateDiffusePureColor(const float3 color, const float3 emissive = float3{0.0f, 0.0f, 0.0f})
    {
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_DIFFUSE;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = color;
        material.texture_.baseTexture_.emissive_ = emissive;
        return material;
    }

    [[nodiscard]] FUNCTION_MODIFIER_STATIC Material CreateLight(const float3 emissive)
    {
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_LIGHT;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = float3{0.0f, 0.0f, 0.0f};
        material.texture_.baseTexture_.emissive_ = emissive;
        return material;
    }

    [[nodiscard]] FUNCTION_MODIFIER_STATIC Material CreateGlassPureColor(
        const float3 color,
        [[maybe_unused]] const float IOR,
        [[maybe_unused]] const float sigmaA,
        [[maybe_unused]] const float abbeNumber = 20.0f,
        [[maybe_unused]] const float tiltNumber = 0.0f)
    {
#ifdef DEBUG_FORCE_ALL_PURE_DIFFUSE
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_DIFFUSE;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = color;
        return material;
#else
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_GLASS;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = color;
        material.texture_.baseTexture_.IOR_ = IOR;
        material.texture_.baseTexture_.sigmaA_ = sigmaA;
        material.texture_.baseTexture_.abbeNumber_ = abbeNumber;
        material.texture_.baseTexture_.tiltNumber_ = tiltNumber;
        return material;
#endif
    }

    [[nodiscard]] FUNCTION_MODIFIER_STATIC Material CreateGGXGlassPureColor(const float3 color, [[maybe_unused]] const float IOR, [[maybe_unused]] const float sigmaA, [[maybe_unused]] const float roughness)
    {
#ifdef DEBUG_FORCE_ALL_PURE_DIFFUSE
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_DIFFUSE;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = color;
        return material;
#else
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_GLASS;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = color;
        material.texture_.baseTexture_.IOR_ = IOR;
        material.texture_.baseTexture_.sigmaA_ = sigmaA;
        material.texture_.baseTexture_.roughness_ = roughness;
        return material;
#endif
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_STATIC Material CreateGGXPureColor(const float3 baseColor,
        [[maybe_unused]] const float roughness, [[maybe_unused]] const float metallic,
        [[maybe_unused]] const float specular, [[maybe_unused]] const float3 emissive = float3{0.0f, 0.0f, 0.0f})
    {
#ifdef DEBUG_FORCE_ALL_PURE_DIFFUSE
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_DIFFUSE;
        material.texture_.textureType_ = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_ = baseColor;
        return material;
#else
        Material material = {};
        material.shadingModel_ = EShadingModel::MAT_GGX;
        material.texture_.textureType_  = ETextureType::TEX_PURE_COLOR;
        material.texture_.baseTexture_.albedo_       = baseColor;
        material.texture_.baseTexture_.roughness_    = clamp(roughness, 0.01f, 1.0f);
        material.texture_.baseTexture_.metallic_     = clamp(metallic, 0.0f, 1.0f);
        material.texture_.baseTexture_.specular_     = clamp(specular, 0.0f, 1.0f);
        material.texture_.baseTexture_.emissive_     = emissive;
        return material;
#endif
    }
};

struct AdditionalLightInfo
{
    int objectIndex_ = -1; // points to SCENE_OBJECTS
    float importanceWeight_ = 0.0f;
    float selectProb_ = 0.0f;
};

struct SDFInfo
{
    float4 sdfInfo0_ = {};
    int cachedVolumeIndex_ = -1;
};

//带UV的顶点和三角形结构体
struct Vertex
{
    float3 position_;
    float3 normal_;
    float2 uv_;
};

struct TriangleInfo
{
    Vertex v0_;
    Vertex v1_;
    Vertex v2_;
};

struct AdditionalObjectInfo
{
    ESDFType sdfType_ = ESDFType::SDF_TORUS;
    SDFInfo sdfInfo_ = {};
    TriangleInfo triangleInfo_ = {};//存储三角形三个顶点数据
};

struct SceneObject
{
    float3 center_;
    float3 extent_;
    Material material_;
    
    float3x3 worldToObject_;
    float3x3 objectToWorld_;
    
    EObjectType type_ = EObjectType::OBJ_CUBE;
    // basically for sdf
    AdditionalObjectInfo additionalObjectInfo_ = {};
    // proxy
    PrimitiveProxy proxy_ = {};
    // dual map
    int objectIndex_ = -1;
    int lightIndex_ = -1;

    bool bVisible_ = true;
    
    FUNCTION_MODIFIER_INLINE SceneObject(): center_({0.0f, 0.0f, 0.0f}), extent_({0.0f, 0.0f, 0.0f}), material_({})
    {
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float GetArea() const
    {
        if(type_ == EObjectType::OBJ_CUBE || type_ == EObjectType::OBJ_SDF)
        {
            const float ex = fabsf(extent_.x);
            const float ey = fabsf(extent_.y);
            const float ez = fabsf(extent_.z);
            return 8.0f * (ex * ey + ex * ez + ey * ez);
        }
        else if(type_ == EObjectType::OBJ_SPHERE)
        {
            const float r = fabsf(extent_.x);
            return 4.0f * CUDART_PI_F * r * r;
        }

        //三角形面积计算
        else if (type_ == EObjectType::OBJ_TRIANGLE)
        {
            const float3 e1 = additionalObjectInfo_.triangleInfo_.v1_.position_ - additionalObjectInfo_.triangleInfo_.v0_.position_;
            const float3 e2 = additionalObjectInfo_.triangleInfo_.v2_.position_ - additionalObjectInfo_.triangleInfo_.v0_.position_;
            return 0.5f * length(cross(e1, e2));
        }

        return 0.0f;
    }
    
    [[nodiscard]] FUNCTION_MODIFIER_INLINE float GetEmissivePower() const
    {
        return luminance(material_.texture_.baseTexture_.emissive_);
    }
    
    FUNCTION_MODIFIER_INLINE void UpdateProxy()
    {
        proxy_.center_ = center_;
        if(type_ == EObjectType::OBJ_CUBE || type_ == EObjectType::OBJ_SDF)
        {
            const float3 axisX = objectToWorld_ * make_float3(1.0f, 0.0f, 0.0f);
            const float3 axisY = objectToWorld_ * make_float3(0.0f, 1.0f, 0.0f);
            const float3 axisZ = objectToWorld_ * make_float3(0.0f, 0.0f, 1.0f);
            float3 half;
            half.x = fabsf(axisX.x) * extent_.x + fabsf(axisY.x) * extent_.y + fabsf(axisZ.x) * extent_.z;
            half.y = fabsf(axisX.y) * extent_.x + fabsf(axisY.y) * extent_.y + fabsf(axisZ.y) * extent_.z;
            half.z = fabsf(axisX.z) * extent_.x + fabsf(axisY.z) * extent_.y + fabsf(axisZ.z) * extent_.z;
            proxy_.boundMin_ = make_float3(center_.x - half.x, center_.y - half.y, center_.z - half.z);
            proxy_.boundMax_ = make_float3(center_.x + half.x, center_.y + half.y, center_.z + half.z);
        }
        else if(type_ == EObjectType::OBJ_SPHERE)
        {
            const float radius = max3(extent_);
            const float3 half = make_float3(radius, radius, radius);
            proxy_.boundMin_ = center_ - half;
            proxy_.boundMax_ = center_ + half;
        }
        //三角形AABB包围盒计算
        else if (type_ == EObjectType::OBJ_TRIANGLE)
        {
            // 将局部坐标顶点经过旋转(objectToWorld_)和平移(center_)转换到世界坐标系
            const float3 p0 = objectToWorld_ * additionalObjectInfo_.triangleInfo_.v0_.position_ + center_;
            const float3 p1 = objectToWorld_ * additionalObjectInfo_.triangleInfo_.v1_.position_ + center_;
            const float3 p2 = objectToWorld_ * additionalObjectInfo_.triangleInfo_.v2_.position_ + center_;

            proxy_.boundMin_ = make_float3(fminf(fminf(p0.x, p1.x), p2.x), fminf(fminf(p0.y, p1.y), p2.y), fminf(fminf(p0.z, p1.z), p2.z));
            proxy_.boundMax_ = make_float3(fmaxf(fmaxf(p0.x, p1.x), p2.x), fmaxf(fmaxf(p0.y, p1.y), p2.y), fmaxf(fmaxf(p0.z, p1.z), p2.z));
        }
    }

    FUNCTION_MODIFIER_INLINE SceneObject(
        const float3 center,
        const float3 extent,
        const Material& material,
        const float3 rotation, // Euler (x,y,z) in angle
        const EObjectType type = EObjectType::OBJ_CUBE,
        const AdditionalObjectInfo& additionalObjectInfo = {})
    : center_(center)
    , extent_(abs(extent))
    , material_(material)
    , type_(type)
    {
        const float3x3 Rx = float3x3::rotX(deg2rad(rotation.x));
        const float3x3 Ry = float3x3::rotY(deg2rad(rotation.y));
        const float3x3 Rz = float3x3::rotZ(deg2rad(rotation.z));

        objectToWorld_ = Rz * (Ry * Rx);
        worldToObject_ = objectToWorld_.transpose();
        additionalObjectInfo_ = additionalObjectInfo;
        UpdateProxy();
    }
};