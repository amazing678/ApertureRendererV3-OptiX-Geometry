// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once


#ifdef LINUX
#undef GUI
#endif
#include "denoise/denoise_statistic.cuh"

#ifdef GUI

#include "camera.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <windows.h>

#include <Shlobj.h>
#pragma comment(lib,"shell32.lib")

inline HWND GL_WINDOW;
inline HWND DX_WINDOW;
inline GLFWwindow* WINDOW = NULL;
inline int ACTUAL_FRAME = 0;

static GLFWwindow* initOpenGL(const int resolutionX, const int resolutionY, const std::string& name)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* Window = glfwCreateWindow(resolutionX, resolutionY, name.c_str(), NULL, NULL);
    if (!Window)
    {
        fprintf(stderr, "Error creating OpenGL WINDOW.\n");;
        glfwTerminate();
    }
    glfwMakeContextCurrent(Window);

    const GLenum result = glewInit();
    if (result != GLEW_OK)
    {
        fprintf(stderr, "GLEW error: %s.\n", reinterpret_cast<const char*>(glewGetErrorString(result)));
        glfwTerminate();
    }

    glfwSwapInterval(0);
    GL_WINDOW = GetActiveWindow();
    return Window;
}

static void initCuda()
{
    int cudaDevices[1];
    unsigned int numCudaDevices;
    cudaGLGetDevices(&numCudaDevices, cudaDevices, 1, cudaGLDeviceListAll);
    if (numCudaDevices == 0)
    {
        fprintf(stderr, "Could not determine CUDA device for current OpenGL context\n.");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(cudaDevices[0]);
}
struct WindowContext
{
    bool movingMouse_;
    double mouseStartX_;
    double mouseStartY_;
    double mouseDX_;
    double mouseDY_;

    double keyW_;
    double keyS_;
    double keyA_;
    double keyD_;
    void clearMouseState()
    {
        mouseDX_ = 0.0f;
        mouseDY_ = 0.0f;
    }
    bool isStateChanging() const
    {
        return mouseDX_ != 0.0 || mouseDY_ != 0.0 ||
            keyW_ != 0.0 || keyS_ != 0.0 || keyA_ != 0.0 || keyD_ != 0.0;
    }
};

// GLFW scroll callback.
static void handleScroll(GLFWwindow* Window, double offsetX, double offsetY)
{
    WindowContext* windowContext = static_cast<WindowContext*>(glfwGetWindowUserPointer(Window));
}

// GLFW keyboard callback.
static void handleKey(GLFWwindow* Window, int key, [[maybe_unused]] int scanCode, int action, [[maybe_unused]] int mods)
{
    WindowContext* windowContext = static_cast<WindowContext*>(glfwGetWindowUserPointer(Window));
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
        {
            glfwSetWindowShouldClose(Window, GLFW_TRUE);
            break;
        }
        case GLFW_KEY_W:
        {
            windowContext->keyW_ = 1.0;
            break;
        }
        case GLFW_KEY_S:
        {
            windowContext->keyS_ = 1.0;
            break;
        }
        case GLFW_KEY_A:
        {
            windowContext->keyA_ = 1.0;
            break;
        }
        case GLFW_KEY_D:
        {
            windowContext->keyD_ = 1.0;
            break;
        }
        default:
        {
            break;
        }
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
        case GLFW_KEY_W:
        {
            windowContext->keyW_ = 0.0;
            break;
        }
        case GLFW_KEY_S:
        {
            windowContext->keyS_ = 0.0;
            break;
        }
        case GLFW_KEY_A:
        {
            windowContext->keyA_ = 0.0;
            break;
        }
        case GLFW_KEY_D:
        {
            windowContext->keyD_ = 0.0;
            break;
        }
        default:
        {
            break;
        }
        }
    }
}

static void handleMouseButton(GLFWwindow* Window, int button, int action, int mods)
{
    WindowContext* windowContext = static_cast<WindowContext*>(glfwGetWindowUserPointer(Window));
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            windowContext->movingMouse_ = true;
            glfwGetCursorPos(Window, &windowContext->mouseStartX_, &windowContext->mouseStartY_);
        }
        else
        {
            windowContext->movingMouse_ = false;
        }
    }
}

static void handleMousePos(GLFWwindow* Window, double posX, double posY)
{
    WindowContext* windowContext = static_cast<WindowContext*>(glfwGetWindowUserPointer(Window));
    if (windowContext->movingMouse_)
    {
        windowContext->mouseDX_ += posX - windowContext->mouseStartX_;
        windowContext->mouseDY_ += posY - windowContext->mouseStartY_;
        windowContext->mouseStartX_ = posX;
        windowContext->mouseStartY_ = posY;
    }
}
static void ResetScreenGBuffer(SceneRenderer* sceneRender, denoise::ScreenGBuffer** screenGBuffer, denoise::ScreenStatisticsBuffer** screenStatisticsBuffer, const int width, const int height)
{
    cudaStreamSynchronize(sceneRender->producerStream_);
    cudaStreamSynchronize(sceneRender->pathGuidingStream_);
    //
    if (*screenGBuffer)
    {
        cudaFree(*screenGBuffer);
        CHECK_ERROR();
    }
    cudaMalloc(screenGBuffer, width * height * sizeof(denoise::ScreenGBuffer));
    CHECK_ERROR();
    cudaMemset(*screenGBuffer, 0, width * height * sizeof(denoise::ScreenGBuffer));
    CHECK_ERROR();
    //
    if (*screenStatisticsBuffer)
    {
        cudaFree(*screenStatisticsBuffer);
        CHECK_ERROR();
    }
    cudaMalloc(screenStatisticsBuffer, width * height * sizeof(denoise::ScreenStatisticsBuffer));
    CHECK_ERROR();
    cudaMemset(*screenStatisticsBuffer, 0, width * height * sizeof(denoise::ScreenStatisticsBuffer));
    CHECK_ERROR();
    //
    if (sceneRender)
    {
        sceneRender->sharedScreenGBufferDevice_ = (*screenGBuffer);
    }
}
static void ResizeBuffers(SceneRenderer* sceneRender, float3** accumBufferCuda, float3** oddBufferCuda, float3** evenBufferCuda, pg::PathGuidingSample** screenSampleBuffer, denoise::ScreenGBuffer** screenGBuffer, denoise::ScreenStatisticsBuffer** screenStatisticsBuffer, cudaGraphicsResource_t* displayBufferCuda, const GLuint tempFB, GLuint* tempTex, const int width, const int height, GLuint displayBuffer)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, displayBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (*displayBufferCuda)
    {
        cudaGraphicsUnregisterResource(*displayBufferCuda);
    }
    cudaGraphicsGLRegisterBuffer(displayBufferCuda, displayBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
    CHECK_ERROR();
    const auto realloc = [width, height](float3** buffer)
        {
            if (*buffer)
            {
                cudaFree(*buffer);
            }
            cudaMalloc(buffer, width * height * sizeof(float3));
            cudaMemset(*buffer, 0, width * height * sizeof(float3));
        };
    realloc(accumBufferCuda);
    realloc(oddBufferCuda);
    realloc(evenBufferCuda);
    CHECK_ERROR();
    {
        if (*screenSampleBuffer)
        {
            cudaFree(*screenSampleBuffer);
            CHECK_ERROR();
        }
        sceneRender->maxRadianceSampleCount_ = width * height * PATH_GUIDING_COLLECT_DEPTH;
        cudaMalloc(screenSampleBuffer, sceneRender->maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample));
        CHECK_ERROR();
        cudaMemset(*screenSampleBuffer, 0, sceneRender->maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample));
        CHECK_ERROR();
        // tell scene sharedCollectedRadianceSampleDevice_
        if (sceneRender)
        {
            sceneRender->sharedCollectedRadianceSampleDevice_ = (*screenSampleBuffer);
        }
        // reset GBuffer
        ResetScreenGBuffer(sceneRender, screenGBuffer, screenStatisticsBuffer, width, height);
    }
    CHECK_ERROR();
    glDeleteTextures(1, tempTex);
    glBindFramebuffer(GL_FRAMEBUFFER, tempFB);
    glGenTextures(1, tempTex);
    glBindTexture(GL_TEXTURE_2D, *tempTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, *tempTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CHECK_ERROR();
}

static GLint addShader(const GLenum shaderType, const char* sourceCode, const GLuint program)
{
    const GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &sourceCode, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    glAttachShader(program, shader);

    return shader;
}

static GLuint createShaderProgram()
{
    GLint success;
    const GLuint program = glCreateProgram();
    static const char* vertexShaderText =
        R"(
#version 330 core
layout(location = 0) in vec3 position;
out vec2 texCoord;
void main()
{
    gl_Position = vec4(position, 1.0);
    texCoord = position.xy * 0.5 + vec2(0.5);
}
)";
    addShader(GL_VERTEX_SHADER, vertexShaderText, program);

    static const char* fragmentShaderText =
        R"(
#version 330 core
in vec2 texCoord;
out vec4 fragColor;
uniform sampler2D texSampler;
void main()
{
    fragColor = texture(texSampler, texCoord);
}
)";
    const GLint fragmentShader = addShader(GL_FRAGMENT_SHADER, fragmentShaderText, program);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        fprintf(stderr, "Error linking shading program\n");
        char info[10240];
        int len;
        glGetShaderInfoLog(fragmentShader, 10240, &len, info);
        fprintf(stderr, info);
        glfwTerminate();
    }
    glUseProgram(program);
    return program;
}

// Create a quad filling the whole screen.
static GLuint createQuad([[maybe_unused]] GLuint program, GLuint* vertexBuffer)
{
    static constexpr float3 vertices[6] =
    {
        { -1.f, -1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        { -1.f,  1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        {  1.f,  1.f, 0.0f },
        { -1.f,  1.f, 0.0f }
    };
    glGenBuffers(1, vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    constexpr GLint posIndex = 0;
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(posIndex, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
    return vertexArray;
}


#include <d3d9.h>
#include <tchar.h>
#include "Imgui/imgui.h"
#include "Imgui/imgui_impl_dx9.h"
#include "Imgui/imgui_impl_win32.h"


class GUIs
{
    ImVec2 nextWindowPos_;

public:
    int changedTimesDesired_ = 0;
    int changedTimesCurrent_ = -1;

    ImVec2 desiredSize_;
    int frameIndex_ = 0;
    int maxFrameCountLog2_ = -1;

    float exposure_ = 1.0f;
    float fps_ = 0.0f;
    float deltaSecond_ = 0.0f;
    float variance_ = 0.0f;

    int toneType_ = 2;
    const Camera& camera_;
    bool pause_ = false;
    char saveName_[100] = "Noname";
    bool needSave_ = false;

    // 🔴 显隐开关
    bool bShowImportedMesh_ = true;

    // 🔴 存储两个渲染核心的指针
    SceneRenderer* rendererBase_;
    SceneRenderer* rendererGun_;

private:
    void DrawMainMenu()
    {
        ImGui::Begin("Settings", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::SetWindowSize(ImVec2(340, 840));
        nextWindowPos_ = ImGui::GetWindowPos() + ImVec2(0, ImGui::GetWindowHeight());
        desiredSize_ = ImGui::GetWindowSize();

        ImGui::Text("Frame No.%d", frameIndex_);
        int frameBufferSizeW;
        int frameBufferSizeH;
        glfwGetFramebufferSize(WINDOW, &frameBufferSizeW, &frameBufferSizeH);
        ImGui::Text("Resolution : %d", frameBufferSizeW);
        ImGui::Text("Render Time: %.3f ms(%.3f FPS)", 1000.f / fps_, fps_);
        ImGui::Text("Variance : %.5f", variance_);
        const float3 cameraPosition = camera_.GetPosition();
        ImGui::Text("Camera Pos: (%.5f, %.5f, %.5f) ", cameraPosition.x, cameraPosition.y, cameraPosition.z);
        const float3 cameraDirection = camera_.GetDirection();
        ImGui::Text("Camera Dir: (%.5f, %.5f, %.5f) ", cameraDirection.x, cameraDirection.y, cameraDirection.z);
        ImGui::Separator();
        ImGui::Text("PathGuiding Min: (%d, %d, %d) ",
            camera_.sceneRenderer_->probeIndirectIndexVolumeStart_.x,
            camera_.sceneRenderer_->probeIndirectIndexVolumeStart_.y,
            camera_.sceneRenderer_->probeIndirectIndexVolumeStart_.z);
        ImGui::Text("PathGuiding Max: (%d, %d, %d) ",
            camera_.sceneRenderer_->probeIndirectIndexVolumeEnd_.x,
            camera_.sceneRenderer_->probeIndirectIndexVolumeEnd_.y,
            camera_.sceneRenderer_->probeIndirectIndexVolumeEnd_.z);
        ImGui::Separator();

        if (pause_)
        {
            if (ImGui::Button("Continue Render")) pause_ = !pause_;
        }
        else
        {
            if (ImGui::Button("Pause Render")) pause_ = !pause_;
        }
        bool changed = false;
        const bool maxFrameCountChanged = ImGui::SliderInt("Max Frame Count (log 2)", &maxFrameCountLog2_, -1, 32);
        changed = maxFrameCountChanged && (frameIndex_ >= (1ull << maxFrameCountLog2_));

        ImGui::Text("Name:"); ImGui::SameLine();
        ImGui::InputText("", saveName_, 100); ImGui::SameLine();
        if (ImGui::Button("Save")) needSave_ = true;

        ImGui::SliderFloat("Camera Exposure", &exposure_, 0, 2);
        if (ImGui::Button("ToneMap")) ImGui::OpenPopup("my_ToneMap_popup");
        ImGui::SameLine();
        ImGui::TextUnformatted(toneType_ == 0 ? "None" : (toneType_ == 1 ? "Gamma" : "ACES"));
        if (ImGui::BeginPopup("my_ToneMap_popup"))
        {
            if (ImGui::Selectable("None")) toneType_ = 0;
            if (ImGui::Selectable("Gamma")) toneType_ = 1;
            if (ImGui::Selectable("ACES")) toneType_ = 2;
            ImGui::EndPopup();
        }

        bool sceneChanged = false;
        sceneChanged |= ImGui::Checkbox("BVH", &camera_.sceneRenderer_->sceneSetting_.bUseBVH_);
        sceneChanged |= ImGui::Checkbox("NEE", &camera_.sceneRenderer_->sceneSetting_.bUseNEE_);
        sceneChanged |= ImGui::Checkbox("PathGuiding", &camera_.sceneRenderer_->sceneSetting_.bUsePathGuiding_);
        sceneChanged |= ImGui::Checkbox("Denoise", &camera_.sceneRenderer_->sceneSetting_.bDenoise_);
        ImGui::Separator();
        sceneChanged |= ImGui::Checkbox("Debug Albedo", &camera_.sceneRenderer_->sceneSetting_.bDebugAlbedo_);
        sceneChanged |= ImGui::Checkbox("Debug Normal", &camera_.sceneRenderer_->sceneSetting_.bDebugNormal_);
        sceneChanged |= ImGui::Checkbox("Debug Depth", &camera_.sceneRenderer_->sceneSetting_.bDebugDepth_);
        sceneChanged |= ImGui::Checkbox("Debug Metallic", &camera_.sceneRenderer_->sceneSetting_.bDebugMetallic_);
        sceneChanged |= ImGui::Checkbox("Debug Roughness", &camera_.sceneRenderer_->sceneSetting_.bDebugRoughness_);
        ImGui::Separator();
        sceneChanged |= ImGui::Checkbox("Debug Statistics Num", &camera_.sceneRenderer_->sceneSetting_.bDebugStatisticsNum_);
        sceneChanged |= ImGui::Checkbox("Debug Statistics Mean", &camera_.sceneRenderer_->sceneSetting_.bDebugStatisticsMean_);
        sceneChanged |= ImGui::Checkbox("Debug Statistics M2", &camera_.sceneRenderer_->sceneSetting_.bDebugStatisticsM2_);
        sceneChanged |= ImGui::Checkbox("Debug Statistics M3", &camera_.sceneRenderer_->sceneSetting_.bDebugStatisticsM3_);

        // ==========================================
        // 🔴 O(1) 零延迟双核热切换！
        // ==========================================
        ImGui::Separator();
        if (ImGui::Checkbox("Show Imported Mesh (Gun)", &bShowImportedMesh_))
        {
            // 1. 保存当前用户的面板设置
            SceneSetting currentSetting = camera_.sceneRenderer_->sceneSetting_;

            // 2. 瞬间切换相机的渲染核心指针！彻底丢掉包含十万个 BVH 节点的旧树！
            const_cast<Camera&>(camera_).sceneRenderer_ = bShowImportedMesh_ ? rendererGun_ : rendererBase_;

            // 3. 将设置同步给新的渲染核心
            const_cast<Camera&>(camera_).sceneRenderer_->sceneSetting_ = currentSetting;

            sceneChanged = true;
        }
        ImGui::Separator();
        // ==========================================

        if (sceneChanged)
        {
            camera_.sceneRenderer_->MarkAsDirty();
            camera_.sceneRenderer_->MarkNeedResetGuiding();
        }
        changed |= sceneChanged;
        if (changed)
        {
            frameIndex_ = 0;
            changedTimesDesired_++;
        }
        ImGui::End();
    }

    void DrawWindowLeftColum(const std::string& name, const int height, bool* show = NULL)
    {
        ImGui::Begin(name.c_str(), show, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        ImGui::SetWindowPos(nextWindowPos_);
        ImGui::SetWindowSize(ImVec2(340.0f, static_cast<float>(height)));

        nextWindowPos_ = nextWindowPos_ + ImVec2(0, ImGui::GetWindowHeight());
        const auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
        desiredSize_ = ImVec2(max(desiredSize_.x, tmp.x), max(desiredSize_.y, tmp.y));
    }

public:
    // 🔴 构造函数接收两个核心
    explicit GUIs(const Camera& camara, SceneRenderer* rBase, SceneRenderer* rGun)
        : frameIndex_(0), camera_(camara), rendererBase_(rBase), rendererGun_(rGun)
    {
    }

    void OnDrawGUI()
    {
        DrawMainMenu();
    }
};


// Data
static LPDIRECT3D9          g_pD3D = NULL;
static LPDIRECT3DDEVICE9        g_pd3dDevice = NULL;
static D3DPRESENT_PARAMETERS    g_d3dpp = {};

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void ResetDevice();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code
inline int MainLoop(GUIs& gui)
{
    // Create application WINDOW
    const WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("VRenderer Controller"), NULL };
    ::RegisterClassEx(&wc);
    RECT rect;
    GetWindowRect(GL_WINDOW, &rect);
    const HWND hwnd = ::CreateWindowEx(/*WS_EX_TOPMOST | */WS_EX_TRANSPARENT | WS_EX_LAYERED, wc.lpszClassName, _T("Controller"), WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME, rect.right + 10, rect.top, 512, 1024, GL_WINDOW, NULL, wc.hInstance, NULL);
    SetLayeredWindowAttributes(hwnd, 0, true, LWA_ALPHA);
    SetLayeredWindowAttributes(hwnd, 0, RGB(0, 0, 0), LWA_COLORKEY);
    LONG_PTR Style = ::GetWindowLongPtr(hwnd, GWL_STYLE);
    Style = Style & ~WS_CAPTION & ~WS_SYSMENU & ~WS_SIZEBOX;
    ::SetWindowLongPtr(hwnd, GWL_STYLE, Style);
    DWORD dwExStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
    dwExStyle &= ~(WS_VISIBLE);
    dwExStyle |= WS_EX_TOOLWINDOW;
    dwExStyle &= ~(WS_EX_APPWINDOW);
    SetWindowLong(hwnd, GWL_EXSTYLE, dwExStyle);
    ShowWindow(hwnd, SW_HIDE);
    UpdateWindow(hwnd);
    DX_WINDOW = hwnd;

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX9_Init(g_pd3dDevice);

    SetWindowLong(hwnd, GWL_EXSTYLE, (GetWindowLong(hwnd, GWL_EXSTYLE) & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED);

    auto& style = ImGui::GetStyle();
    style.FrameRounding = 12.f;
    style.GrabRounding = 12.f;

    bool first = true;
    // Main loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (msg.message != WM_QUIT)
    {
        if (GL_WINDOW == 0) break;

        if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX9_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        gui.OnDrawGUI();

        // Rendering
        ImGui::EndFrame();

        g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
        if (g_pd3dDevice->BeginScene() >= 0)
        {
            ImGui::Render();
            ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
            g_pd3dDevice->EndScene();
        }

        HRESULT result = g_pd3dDevice->Present(NULL, NULL, NULL, NULL);

        if (first)
        {
            ShowWindow(hwnd, SW_SHOWDEFAULT);
            UpdateWindow(hwnd);
            RECT rect;
            GetWindowRect(DX_WINDOW, &rect);
            GetWindowRect(GL_WINDOW, &rect);
            float3 aim_ = float3{ (float)rect.right + 10, (float)rect.top, 0 };
            first = false;
        }

        //Handle loss of D3D9 device
        if (result == D3DERR_DEVICELOST && g_pd3dDevice->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
        {
            ResetDevice();
        }

        RECT rect;
        RECT rect2;
        GetWindowRect(GL_WINDOW, &rect);
        GetWindowRect(DX_WINDOW, &rect2);

        const float3 desiredLocation = float3{ static_cast<float>(rect.right) + 10, static_cast<float>(rect.top), 0 };
        MoveWindow(DX_WINDOW, static_cast<int>(desiredLocation.x), static_cast<int>(desiredLocation.y), static_cast<int>(gui.desiredSize_.x), static_cast<int>(gui.desiredSize_.y), FALSE);

        static bool active = false;
        if (GetForegroundWindow() == GL_WINDOW)
        {
            if (active == false)
            {
                active = true;
            }
            [[maybe_unused]] auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
        }
        else
        {
            active = false;
        }
    }
    ImGui_ImplDX9_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions

inline bool CreateDeviceD3D(const HWND hwnd)
{
    if ((g_pD3D = Direct3DCreate9(D3D_SDK_VERSION)) == NULL)
    {
        return false;
    }
    // Create the D3DDevice
    ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));
    g_d3dpp.Windowed = TRUE;
    g_d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    g_d3dpp.BackBufferFormat = D3DFMT_UNKNOWN; // Need to use an explicit format with alpha if needing per-pixel alpha composition.
    g_d3dpp.EnableAutoDepthStencil = TRUE;
    g_d3dpp.AutoDepthStencilFormat = D3DFMT_D16;
    g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_ONE;            // Present with vsync
    //g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;   // Present without vsync, maximum unthrottled framerate
    if (g_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &g_d3dpp, &g_pd3dDevice) < 0)
        return false;

    return true;
}

inline void CleanupDeviceD3D()
{
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
    if (g_pD3D) { g_pD3D->Release(); g_pD3D = NULL; }
}

inline void ResetDevice()
{
    ImGui_ImplDX9_InvalidateDeviceObjects();
    HRESULT hr = g_pd3dDevice->Reset(&g_d3dpp);
    if (hr == D3DERR_INVALIDCALL)
    {
        IM_ASSERT(0);
    }
    ImGui_ImplDX9_CreateDeviceObjects();
}

// Win32 message handler
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
    {
        return true;
    }
    if (msg == WM_SETFOCUS)
    {
        [[maybe_unused]] auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
        RECT rect;
        GetWindowRect(GL_WINDOW, &rect);
        SetWindowPos(GL_WINDOW, DX_WINDOW, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_NOACTIVATE);
    }
    switch (msg)
    {
    case WM_SIZE:
    {
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            g_d3dpp.BackBufferWidth = LOWORD(lParam);
            g_d3dpp.BackBufferHeight = HIWORD(lParam);
            ResetDevice();
        }
        return 0;
    }
    case WM_SYSCOMMAND:
    {
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
        {
            return 0;
        }
        break;
    }
    case WM_DESTROY:
    {
        ::PostQuitMessage(0);
        return 0;
    }
    default:
    {
        break;
    }
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}


inline void WriteBitmapFile(char* filename, int width, int height, unsigned char* bitmapData)
{
    BITMAPFILEHEADER bitmapFileHeader = {};
    bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;   //BM
    bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    BITMAPINFOHEADER bitmapInfoHeader = {};
    bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfoHeader.biWidth = width;
    bitmapInfoHeader.biHeight = height;
    bitmapInfoHeader.biPlanes = 1;
    bitmapInfoHeader.biBitCount = 24;
    bitmapInfoHeader.biCompression = BI_RGB;
    bitmapInfoHeader.biSizeImage = width * abs(height) * 3;

    //swap R B
    for (unsigned int imageIdx = 0u; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3u)
    {
        const unsigned char tempRGB = bitmapData[imageIdx];
        bitmapData[imageIdx] = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }
    char bmpFileName[128];
    sprintf(bmpFileName, "%s.bmp", filename);
    FILE* filePtr = fopen(bmpFileName, "wb");

    fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
    fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);
    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);
    fclose(filePtr);
}

inline void SaveFrameBuffer(GLFWwindow* window, char* fileName)
{
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);

    GLint lineWidth = width * 3;
    lineWidth = (lineWidth + 3) / 4 * 4;

    const GLint pixelDataLength = lineWidth * height;

    GLubyte* pixelDataPtr = static_cast<GLubyte*>(malloc(pixelDataLength));
    if (pixelDataPtr == 0)
    {
        exit(0);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixelDataPtr);
    WriteBitmapFile(fileName, lineWidth / 3, height, pixelDataPtr);
    free(pixelDataPtr);
}
#endif // GUI

// 🔴 最终修改版：接收两个核心！
inline void RunGUI(Camera& camara, SceneRenderer* rendererBase, SceneRenderer* rendererGun, const bool hideConsole = false)
{
    std::cout << "GUI Initializing." << std::endl;
#if GUI
    if (hideConsole)
    {
        HWND hwnd;
        hwnd = FindWindow("ConsoleWindowClass", NULL);
        if (hwnd)
        {
            ShowOwnedPopups(hwnd, SW_HIDE);
            ShowWindow(hwnd, SW_HIDE);
        }
    }
    std::cout << "Console Initialized." << std::endl;

    // 🔴 把两个核心喂给 GUI 面板
    GUIs gui(camara, rendererBase, rendererGun);
    std::cout << "GUI Initialized." << std::endl;

    WindowContext windowContext = {};
    std::cout << "WindowContext Initialized." << std::endl;

    GLuint displayBuffer = 0;
    GLuint displayTex = 0;
    GLuint program = 0;
    GLuint quadVertexBuffer = 0;
    GLuint quadVao = 0;
    int width = -1;
    int height = -1;

    // Init OpenGL WINDOW and callbacks.
    WINDOW = initOpenGL(camara.resolutionX_, camara.resolutionY_, camara.name_);
    std::cout << "GL Initialized." << std::endl;
    glfwSetWindowUserPointer(WINDOW, &windowContext);
    glfwSetKeyCallback(WINDOW, handleKey);
    glfwSetScrollCallback(WINDOW, handleScroll);
    glfwSetCursorPosCallback(WINDOW, handleMousePos);
    glfwSetMouseButtonCallback(WINDOW, handleMouseButton);

    glGenBuffers(1, &displayBuffer);
    glGenTextures(1, &displayTex);

    GLuint tempBuffer = 0;
    GLuint tempTex = 0;
    glGenFramebuffers(1, &tempBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, tempBuffer);
    glGenTextures(1, &tempTex);

    glBindTexture(GL_TEXTURE_2D, tempTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, camara.resolutionX_, camara.resolutionY_, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tempTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    program = createShaderProgram();
    quadVao = createQuad(program, &quadVertexBuffer);
    std::cout << "GL Prepared." << std::endl;

    initCuda();
    std::cout << "Cuda Initialized." << std::endl;
    float3* accumBuffer_ = NULL;
    float3* oddAccumBuffer_ = NULL;
    float3* evenAccumBuffer_ = NULL;
    // resX * resY * PATH_GUIDING_COLLECT_DEPTH
    pg::PathGuidingSample* collectedRadianceSampleDevice_ = nullptr;
    denoise::ScreenGBuffer* screenGBufferDevice_ = nullptr;
    denoise::ScreenStatisticsBuffer* screenStatisticsBufferDevice_ = nullptr;

    cudaGraphicsResource_t displayBufferCuda_ = NULL;

    auto panel = std::thread([&]()
        {
            MainLoop(gui);
        });
    while (!glfwWindowShouldClose(WINDOW))
    {
        // Process events.
        glfwPollEvents();
        WindowContext* windowContext = static_cast<WindowContext*>(glfwGetWindowUserPointer(WINDOW));
        if (windowContext->isStateChanging() || gui.changedTimesDesired_ != gui.changedTimesCurrent_)
        {
            CHECK_ERROR();
            // 🔴 此时清理当前相机的渲染器缓冲
            ResetScreenGBuffer(gui.camera_.sceneRenderer_, &screenGBufferDevice_, &screenStatisticsBufferDevice_, width, height);
            CHECK_ERROR();
            camara.UpdateCamera(windowContext->mouseDX_, windowContext->mouseDY_, (windowContext->keyW_ - windowContext->keyS_) * gui.deltaSecond_, (windowContext->keyA_ - windowContext->keyD_) * gui.deltaSecond_);
            CHECK_ERROR();
            windowContext->clearMouseState();
            gui.frameIndex_ = 0;
            gui.changedTimesCurrent_ = gui.changedTimesDesired_;
        }

        // Reallocate buffers if WINDOW size changed.
        int newWidth;
        int newHeight;
        glfwGetFramebufferSize(WINDOW, &newWidth, &newHeight);
        if (newWidth != width || newHeight != height)
        {
            width = newWidth;
            height = newHeight;
            // 🔴 同上，自适应使用当前的 sceneRenderer
            ResizeBuffers(gui.camera_.sceneRenderer_, &accumBuffer_, &oddAccumBuffer_, &evenAccumBuffer_, &collectedRadianceSampleDevice_, &screenGBufferDevice_, &screenStatisticsBufferDevice_, &displayBufferCuda_, tempBuffer, &tempTex, width, height, displayBuffer);
            CHECK_ERROR();
            glViewport(0, 0, width, height);
            gui.frameIndex_ = 0;
            // Allocate texture once
            glBindTexture(GL_TEXTURE_2D, displayTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        // Map GL buffer for access with CUDA.
        cudaGraphicsMapResources(1, &displayBufferCuda_, /*stream=*/0);
        void* p;
        size_t size_p;
        cudaGraphicsResourceGetMappedPointer(&p, &size_p, displayBufferCuda_);

        if (!gui.pause_ && (gui.maxFrameCountLog2_ < 0 || gui.frameIndex_ < (1ull << gui.maxFrameCountLog2_)))
        {
            CHECK_ERROR();
            // 🔴 这里也改成了调用相机的当前渲染器
            camara.sceneRenderer_->SetExposure(gui.exposure_);
            CHECK_ERROR();
            auto startTime = std::chrono::system_clock::now();
            // real render entry
            camara.Render(
                accumBuffer_,
                oddAccumBuffer_,
                evenAccumBuffer_,
                gui.camera_.sceneRenderer_->sceneSetting_.bUsePathGuiding_ ? collectedRadianceSampleDevice_ : nullptr,
                gui.camera_.sceneRenderer_->sceneSetting_.bDenoise_ ? screenGBufferDevice_ : nullptr,
                gui.camera_.sceneRenderer_->sceneSetting_.bDenoise_ ? screenStatisticsBufferDevice_ : nullptr,
                static_cast<unsigned int*>(p),
                int2{ width , height },
                gui.frameIndex_,
                gui.toneType_
            );

            auto finishTime = std::chrono::system_clock::now();
            gui.deltaSecond_ = std::chrono::duration<float>(finishTime - startTime).count();
            const float newFPS = 1.0f / gui.deltaSecond_;
            const float interpAlpha = 1.0f - expf(-gui.deltaSecond_ * 30.0f); // 0.1 -> 1 - exp(-3), 1 -> 1 - exp(-30)
            gui.fps_ = lerp(gui.fps_, newFPS, abs(newFPS - gui.fps_) / gui.fps_ > 0.3f ? 1.0f : interpAlpha);
            gui.variance_ = static_cast<float>(lerpd(gui.variance_, camara.variance_, abs(camara.variance_ - gui.variance_) / (gui.variance_ + 1.0e-4f) > 0.3f ? 1.0f : interpAlpha));
            gui.frameIndex_++;
            ACTUAL_FRAME++;
        }

        // Unmap GL buffer.
        cudaGraphicsUnmapResources(1, &displayBufferCuda_, /*stream=*/0);

        // Update texture for display.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, displayBuffer);
        glBindTexture(GL_TEXTURE_2D, displayTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Render the quad.
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(quadVao);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(WINDOW);

        if (gui.needSave_)
        {
            printf("saved.\n");
            SaveFrameBuffer(WINDOW, gui.saveName_);
            gui.needSave_ = false;
        }
    }

    cudaFree(accumBuffer_);
    cudaFree(oddAccumBuffer_);
    cudaFree(evenAccumBuffer_);
    cudaFree(collectedRadianceSampleDevice_);
    cudaFree(screenGBufferDevice_);
    cudaFree(screenStatisticsBufferDevice_);

    // Cleanup OpenGL.
    glDeleteVertexArrays(1, &quadVao);
    glDeleteBuffers(1, &quadVertexBuffer);
    glDeleteProgram(program);
    glfwDestroyWindow(WINDOW);
    glfwTerminate();

    GL_WINDOW = 0;

    panel.join();

#else
    printf("GUI has not been Compiled.\n");
#endif // GUI
}