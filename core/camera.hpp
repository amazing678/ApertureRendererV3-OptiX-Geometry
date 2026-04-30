// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "scene.hpp"
#include <functional>
#include "tonemap.cuh"
#include "denoise/denoise.cuh"
#include "denoise/denoise_statistic.cuh"

typedef std::function<float3(const float3 radiance)> ToneMapFunction;

int SaveBMP(unsigned char* image, int imageWidth, int imageHeight, const char* filename);

class Camera 
{
protected:
    float3 position_;
    float3 direction_;

public:
    SceneRenderer* sceneRenderer_;
    
public:
    int resolutionX_;
    int resolutionY_;
    std::string name_;

    mutable double variance_ = 0.0f;

    void SetPosition(const float3 position)
    {
        position_ = position;
    }
    float3 GetPosition() const
    {
        return position_;
    }
    void SetDirection(const float3 direction)
    {
        direction_ = normalize(direction);
    }
    float3 GetDirection() const
    {
        return direction_;
    }
    void SetPositionAndDirection(const float3 position, const float3 direction)
    {
        SetPosition(position);
        SetDirection(direction);
    }
    void SetVolume(SceneRenderer& sceneRenderer)
    {
        this->sceneRenderer_ = &sceneRenderer;
    }

    Camera(SceneRenderer& sceneRenderer, const std::string& name = "test camera", const int resolutionX = 1080, int resolutionY = 720, const float3 position = { 0.0f, 0.75f, 0.75f }, const float3 direction = { 0.0f, 0.0f, -0.75f })
    {
        SetPositionAndDirection(position, direction);
        this->name_ = name;
        this->sceneRenderer_ = &sceneRenderer;
        this->resolutionX_ = resolutionX;
        this->resolutionY_ = resolutionY;
    }
    
    void UpdateCamera(const double deltaMouseX, const double deltaMouseY, const double deltaWS, const double deltaAD)
    {
        constexpr float PITCH_LIMIT_DEG  = 89.0f;
        constexpr float MOUSE_X_SCALE       = 0.0025f;
        constexpr float MOUSE_Y_SCALE       = 0.0025f;
        constexpr float MOVE_FORWARD_SPEED = 0.5f;
        constexpr float MOVE_RIGHT_SPEED   = 0.5f;
        constexpr float DEG2RAD = 3.1415926f / 180.0f;

        const float3 camaraDirection = GetDirection();
        float yaw   = atan2f(camaraDirection.x, camaraDirection.z);
        float pitch = asinf(fminf(1.f, fmaxf(-1.f, camaraDirection.y)));
        yaw   -= static_cast<float>(deltaMouseX) * MOUSE_X_SCALE;
        pitch -= static_cast<float>(deltaMouseY) * MOUSE_Y_SCALE;

        const float pitchLimit = PITCH_LIMIT_DEG * DEG2RAD;
        pitch = fminf(+pitchLimit, fmaxf(-pitchLimit, pitch));

        const float cp = cosf(pitch);
        const float sp = sinf(pitch);
        const float sy = sinf(yaw);
        const float cy = cosf(yaw);
        const float3 forward = make_float3(sy * cp, sp, cy * cp);
        const float3 worldUp = make_float3(0.f, 1.f, 0.f);
        float3 right = cross(worldUp, forward);

        const float3 deltaPosition =
            forward * (MOVE_FORWARD_SPEED * static_cast<float>(deltaWS)) +
            right   * (MOVE_RIGHT_SPEED   * static_cast<float>(deltaAD));
        SetPositionAndDirection(GetPosition() + deltaPosition, forward);
    }
    
    void RenderToFile(const std::string& path, const int sampleNum = 1, const ToneMapFunction tonemapFunction = ACES, const float exp = 1.0) const;

    void Render(float3* target, float3* oddTarget, float3* evenTarget, pg::PathGuidingSample* sampleTarget, denoise::ScreenGBuffer* screenGBuffer, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer, unsigned int* target2, const int2 size, const int frame, const int tone = 2) const;
};