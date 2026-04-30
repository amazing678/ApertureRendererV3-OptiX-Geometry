// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#include "random.cuh"
#include "scene.hpp"
#include "render/color.cuh"
#include "tiny_obj_loader.h"

// 引入读取外部文件的基础库
#include <stdio.h>
#include <string>

struct SceneHelper
{
    constexpr static inline float THICK = 0.01f;
    inline static std::vector<SceneObject> CreateCornellBox(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float3 wallAColor = color::Red(),
        const float3 wallBColor = color::Green(),
        const float3 lightColor = color::White() * 10.0f,
        const float lightSize = 0.25f, const float3 lightOffset = {0.0f, 0.0f, 0.0f}, const bool multipleLight = false)
    {
        std::vector<SceneObject> Scene;
        // down
        Scene.emplace_back(
            float3{ 0.0f,-0.5f - THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.25f), 0.05f, 0.0f, 0.5f), // Color::WhiteDarken(0.25f)
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // up
        Scene.emplace_back(
            float3{ 0.0f,0.5f + THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // light
        if(multipleLight)
        {
            Scene.emplace_back(
                float3{ 0.25f, 0.5f - THICK, 0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ -0.25f, 0.5f - THICK, 0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ 0.25f, 0.5f - THICK, -0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ -0.25f, 0.5f - THICK, -0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            
        }
        else
        {
            Scene.emplace_back(
                float3{ 0.0f,0.5f - THICK,0.0f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
        }
        // back
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.5f - THICK } + position, float3{ 0.5f,0.5f,THICK },
            Material::CreateGGXPureColor(color::WhiteDarken(0.25f), 0.25f, 1.0f, 0.5f),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );

        Scene.emplace_back(
            float3{ -0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallAColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallBColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    inline static std::vector<SceneObject> CreateDiffuseCornellBox(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float3 wallAColor = color::Red(),
        const float3 wallBColor = color::Green(),
        const float3 lightColor = color::White() * 10.0f,
        const float lightSize = 0.25f)
    {
        std::vector<SceneObject> Scene;
        // down
        Scene.emplace_back(
            float3{ 0.0f,-0.5f - THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // up
        Scene.emplace_back(
            float3{ 0.0f,0.5f + THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // light
        Scene.emplace_back(
            float3{ 0.0f,0.5f - THICK,0.0f } + position, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
            Material::CreateLight(lightColor),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // back
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.5f - THICK } + position, float3{ 0.5f,0.5f,THICK },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );

        Scene.emplace_back(
            float3{ -0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallAColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallBColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateSpectrumTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f})
    {
        std::vector<SceneObject> Scene;
        constexpr int WIDTH = 6;
        constexpr int WIDTH_HALF = WIDTH / 2;
        constexpr float SCALE = 0.6f;
        for(int x=-WIDTH_HALF; x<=WIDTH_HALF; x++)
        {
            for(int y=-WIDTH_HALF; y<=WIDTH_HALF; y++)
            {
                for(int z=-WIDTH_HALF; z<=WIDTH_HALF; z++)
                {
                    const float3 center = float3{(static_cast<float>(x) + static_cast<float>(z) / WIDTH) / WIDTH_HALF, (static_cast<float>(y) + static_cast<float>(x) / WIDTH) / WIDTH_HALF, static_cast<float>(z) / WIDTH_HALF} * 0.5f * SCALE;
                    const float abbe = lerp(50.0f, 1.0f, (static_cast<float>(z + WIDTH_HALF) / WIDTH));
                    const float tilt = lerp(1.0f, -1.0f, (static_cast<float>(x + WIDTH_HALF) / WIDTH));
                    const float ior = lerp(0.0f, 1.0f, (static_cast<float>(y + WIDTH_HALF) / WIDTH));
                    Scene.emplace_back(
                        center + position, float3{ 0.045f,0.1f,0.045f } * SCALE * 4 / WIDTH,
                        Material::CreateGlassPureColor(color::WhiteDarken(0.5f), lerp(1.3f, 2.5f, ior), 10.0f, abbe, tilt), 
                        float3{0.0f, 0.0f, 0.0f},
                        EObjectType::OBJ_CUBE
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0f, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass ball 1
        Scene.emplace_back(
            float3{ 0.0f,0.25f,0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.6f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // test light
        Scene.emplace_back(
            float3{ 0.125f,0.25f,-0.125f } + position, float3{ 0.05f,0.05f,0.05f },
            Material::CreateLight(color::White() * 25.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 2
        Scene.emplace_back(
            float3{ -0.25f,0.0f,-0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::Yellow(), lerp(1.0f, 1.5f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 3
        Scene.emplace_back(
            float3{ 0.25f,0.0f,0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::Cyan(), lerp(1.0, 1.55f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 4
        Scene.emplace_back(
            float3{ 0.0,0.0,0.0 } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::RedLighten(0.01f), lerp(1.0f, 1.45f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 5
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,-0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::BlueLighten(0.01f), lerp(1.0f, 1.5f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass plane 1
        Scene.emplace_back(
            float3{ 0.0f,0.0f,0.25f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::Azure(), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // glass plane 2
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::RedLighten(0.25f), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // glass plane 3
        Scene.emplace_back(
            float3{ 0.0f,0.25f,0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.25f), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{-30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 1
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.25f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::Orange(), 0.1f, 1.0f, 1.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 2
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,-0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.9f), 0.1f, 0.0f, 0.5f), 
            float3{-30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 3
        Scene.emplace_back(
            float3{ 0.0f,0.25f,-0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.5f), 0.5f, 1.0f, 1.0f), 
            float3{30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateBigGlassTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0)
    {
        std::vector<SceneObject> Scene;
        // test light
        /*
        Scene.emplace_back(
            float3{ 0.125,0.25,-0.125 } + position, float3{ 0.05f,0.05f,0.05f },
            Material::CreateLight(Color::White() * 25.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        */
        // glass ball
        Scene.emplace_back(
            float3{ -0.15f,0.0f,-0.2f } + position, float3{ 0.25f,0.25f,0.25f },
            Material::CreateGlassPureColor(color::CyanLighten(0.2f), lerp(1.0f, 1.45f, IORStrength), 5.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass cube
        Scene.emplace_back(
            float3{ 0.15f,-0.2f,0.15f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.2f), lerp(1.0f, 1.45f, IORStrength), 5.0f), 
            float3{15.0f, 30.0f, 70.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateCubeTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass cube
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ -0.2f,-0.2f,0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ -0.2f,-0.2f,-0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::CyanLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,-0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::YellowLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateAbbeTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass sphere
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,0.1f } + position, float3{ 0.2f,0.2f,0.2f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass cube
        Scene.emplace_back(
            float3{ -0.1f,0.1f,-0.15f } + position, float3{ 0.1f,0.25f,0.1f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{15.0f, 30.0f, 70.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTestScene2(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2,
        const float3 Color = color::Red(),
        const EObjectType objectType = EObjectType::OBJ_SPHERE,
        const float randomDiffuseRate = 0.0f)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const auto currentColor = lerp(Color, color::White(), normalized.x);
                    Scene.emplace_back(
                        location, float3{ 0.025f,0.025f,0.025f } * scale * objectScale,
                        random < randomDiffuseRate ? Material::CreateDiffusePureColor(currentColor) : Material::CreateGlassPureColor(currentColor, 1.1f + normalized.y, 1.0f + normalized.z * 100.0f), 
                        float3{0.0f, 0.0f, 0.0f},
                        objectType
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTorus(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const float random2 = Hash11f(static_cast<uint32_t>(random * INT_MAX));
                    const float random3 = Hash11f(static_cast<uint32_t>(random2 * INT_MAX));
                    const float random4 = Hash11f(static_cast<uint32_t>(random3 * INT_MAX));
                    const float random5 = Hash11f(static_cast<uint32_t>(random4 * INT_MAX));
                    const float random6 = Hash11f(static_cast<uint32_t>(random5 * INT_MAX));
                    const float3 color = color::HSV2RGB(random6, 1.0f, 1.0f);
                    const auto currentColor = lerp(color, color::White(), normalized.x * 0.7f);
                    Scene.emplace_back(
                        location, float3{ 0.05f,0.05f,0.05f } * scale * objectScale,
                        Material::CreateGlassPureColor(currentColor, 1.3f + 2.0f * normalized.y, 5.0f + normalized.z * 50.0f, 1.0f, 0.1f), 
                        float3{random2 * 360.0f, random3 * 360.0f, random4 * 360.0f},
                        EObjectType::OBJ_SDF,
                        sdf::CreateTorus()
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassSDFVolume(
        const int volumeIndex,
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const float random2 = Hash11f(static_cast<uint32_t>(random * INT_MAX));
                    const float random3 = Hash11f(static_cast<uint32_t>(random2 * INT_MAX));
                    const float random4 = Hash11f(static_cast<uint32_t>(random3 * INT_MAX));
                    const float random5 = Hash11f(static_cast<uint32_t>(random4 * INT_MAX));
                    const float random6 = Hash11f(static_cast<uint32_t>(random5 * INT_MAX));
                    const float3 color = color::HSV2RGB(random6, 1.0f, 1.0f);
                    const auto currentColor = lerp(color, color::White(), normalized.z * 0.7f);
                    Scene.emplace_back(
                        location, float3{ 0.05f,0.05f,0.05f } * scale * objectScale,
                        Material::CreateGlassPureColor(currentColor, 1.6f + 2.0f * normalized.y, 5.0f + normalized.x * 25.0f, 1.0f, 0.1f), 
                        float3{random2 * 360.0f, random3 * 360.0f, random4 * 360.0f},
                        EObjectType::OBJ_SDF,
                        sdf::CreateSDFVolume(volumeIndex)
                    );
                }
            }
        }
        return Scene;
    }

    //加载带纹理的三角形网格
    static inline void LoadTexturedMesh(
        int objectTypeID,
        const std::string& objPath,
        const float3 position,
        const float scale,
        cudaTextureObject_t albedoTex,
        cudaTextureObject_t normalTex,
        cudaTextureObject_t metallicTex,
        cudaTextureObject_t roughnessTex,
        std::vector<SceneObject>& outSceneObjects)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str()))
        {
            std::cerr << "Failed to load OBJ: " << err << std::endl;
            return;
        }

        for (const auto& shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) continue; // 确保是三角形

                /*SceneObject triObj(
                    position,
                    make_float3(scale, scale, scale),
                    Material().CreateGGXPureColor(color().White(), 0.4f, 0.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 0.0f),
                    EObjectType::OBJ_TRIANGLE
                );*/

                SceneObject triObj(
                    position,
                    make_float3(scale, scale, scale),
                    Material::CreateGGXPureColor(color::White(), 0.4f, 0.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 0.0f),
                    EObjectType::OBJ_TRIANGLE
                );

                //挂载贴图句柄
                triObj.material_.texture_.baseTexture_.albedoTex_ = albedoTex;
                triObj.material_.texture_.baseTexture_.normalTex_ = normalTex;
                triObj.material_.texture_.baseTexture_.metallicTex_ = metallicTex;
                triObj.material_.texture_.baseTexture_.roughnessTex_ = roughnessTex; 

                // 提取三角形的三个顶点
                Vertex vertices[3];
                for (size_t v = 0; v < 3; v++)
                {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    // 顶点位置 (缩放)
                    vertices[v].position_ = make_float3(
                        attrib.vertices[3 * size_t(idx.vertex_index) + 0] * scale,
                        attrib.vertices[3 * size_t(idx.vertex_index) + 1] * scale,
                        attrib.vertices[3 * size_t(idx.vertex_index) + 2] * scale
                    );

                    // 顶点法线
                    if (idx.normal_index >= 0) {
                        vertices[v].normal_ = make_float3(
                            attrib.normals[3 * size_t(idx.normal_index) + 0],
                            attrib.normals[3 * size_t(idx.normal_index) + 1],
                            attrib.normals[3 * size_t(idx.normal_index) + 2]
                        );
                    }
                    else {
                        vertices[v].normal_ = { 0.0f, 1.0f, 0.0f };
                    }

                    // 顶点 UV (注意翻转 V 轴，图形学常态)
                    if (idx.texcoord_index >= 0) {
                        vertices[v].uv_ = make_float2(
                            attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                            1.0f - attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]
                        );
                    }
                    else {
                        vertices[v].uv_ = { 0.0f, 0.0f };
                    }
                }

                // 如果没有提供法线，自动算一个面法线
                if (shape.mesh.indices[index_offset].normal_index < 0)
                {
                    float3 e1 = vertices[1].position_ - vertices[0].position_;
                    float3 e2 = vertices[2].position_ - vertices[0].position_;
                    float3 n = normalize(cross(e1, e2));
                    vertices[0].normal_ = n; vertices[1].normal_ = n; vertices[2].normal_ = n;
                }

                triObj.additionalObjectInfo_.triangleInfo_.v0_ = vertices[0];
                triObj.additionalObjectInfo_.triangleInfo_.v1_ = vertices[1];
                triObj.additionalObjectInfo_.triangleInfo_.v2_ = vertices[2];

                // 计算该三角形的 AABB
                triObj.UpdateProxy();

                // 将做好的三角形塞进场景大数组
                outSceneObjects.push_back(triObj);
                index_offset += fv;
            }
        }
        std::cout << "Loaded textured mesh: " << objPath << " (" << outSceneObjects.size() << " triangles total in scene)" << std::endl;
    }


    // 1. 读取 Bin 格式体素数据的加载器
    inline static float* LoadSDFVolumeToGPU(const std::string& filepath, int resolution) {
        size_t numElements = resolution * resolution * resolution;
        size_t byteSize = numElements * sizeof(float);

        FILE* file = fopen(filepath.c_str(), "rb");
        if (!file) {
            printf("[错误] 无法打开 SDF 模型文件: %s\n", filepath.c_str());
            return nullptr;
        }

        std::vector<float> cpuData(numElements);
        fread(cpuData.data(), sizeof(float), numElements, file);
        fclose(file);

        float* d_volumeData = nullptr;
        cudaMalloc(&d_volumeData, byteSize);
        cudaMemcpy(d_volumeData, cpuData.data(), byteSize, cudaMemcpyHostToDevice);

        printf("成功加载外部模型至显存: %s\n", filepath.c_str());
        return d_volumeData;
    }

    // 2. 构建包含外部导入模型的专属场景
    inline static std::vector<SceneObject> CreateCustomSDFScene(SceneRenderer& renderer, const float3& center) {
        // 创建一个经典的纯白康奈尔盒子作为打光环境
        std::vector<SceneObject> Scene = CreateCornellBox(center, color::White(), color::White(), color::White() * 15.0f, 0.5f);

        // ------------------- 配置区 -------------------
        const int SDF_RESOLUTION = 128; // 必须和 Blender/Houdini 脚本里设置的一模一样
        const std::string SDF_PATH = "D:/blender_sdf_128.bin"; // 换成你实际导出的文件路径！
        // ----------------------------------------------

        float* sdfVolumeGPU = LoadSDFVolumeToGPU(SDF_PATH, SDF_RESOLUTION);

        if (sdfVolumeGPU) {
            sdf::SDFVolumeInfo volumeInfo;
            volumeInfo.volumePtr_ = sdfVolumeGPU;
            volumeInfo.dim_ = make_int3(SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION);
            volumeInfo.ComputeVoxelSize();

            AdditionalObjectInfo customSDFInfo = sdf::CreateSDFVolume(0);

            renderer.AddVolumeToCacheManager(
                { 64, 64, 64 },
                { 0.0f, 0.0f, 0.0f },
                { 1.0f, 1.0f, 1.0f },
                sdf::SDFVolumeFunctor(volumeInfo),
                customSDFInfo.sdfInfo_
            );

            // 放进场景中
            Scene.emplace_back(
                center + make_float3(0.0f, 0.5f, 0.0f),
                make_float3(1.2f, 1.2f, 1.2f),
                Material::CreateGlassPureColor(color::White(), 1.5f, 15.0f, 0.01f),
                make_float3(0.0f, 0.0f, 0.0f),
                EObjectType::OBJ_SDF,
                customSDFInfo
            );
        }
        else {
            // 如果没找到文件，放个占位球防止画面全黑
            Scene.emplace_back(
                center + make_float3(0.0f, 0.5f, 0.0f),
                make_float3(0.3f, 0.3f, 0.3f), // 必须是 float3 的缩放
                Material::CreateGlassPureColor(color::White(), 1.5f, 15.0f, 0.05f),
                make_float3(0.0f, 0.0f, 0.0f), // 必须补上 float3 的旋转
                EObjectType::OBJ_SPHERE
            );
        }

        return Scene;
    }
};
