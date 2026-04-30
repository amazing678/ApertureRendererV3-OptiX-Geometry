#include "scene.hpp"
#include "camera.hpp"
#include "helper/scene_helper.h"

//引入图片和模型解析库
#define STB_IMAGE_IMPLEMENTATION
#include "core/helper/texture_loader.cuh"
#define TINYOBJLOADER_IMPLEMENTATION
#include "core/helper/tiny_obj_loader.h"

#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <tensor/tensor.cuh>
#include "spectrum//sample_ciexyz.cuh"
#include "spectrum/spectrum_lut.cuh"

std::vector<SceneObject> CreateScene()
{
    std::vector<SceneObject> Scene;
    /*
    std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f},
        Color::Red(),
        Color::Green(),
        Color::YellowLighten(0.9f) * 250.0f, 0.1f, {0.4f, -0.75f, 0.4f});
    Scene.insert(Scene.end(), CornellBox1.begin(), CornellBox1.end());
    Scene.emplace_back(
        float3{ 0.0f,-0.1f,0.0f }, float3{ 0.1f,0.4f,0.5f },
        Material::CreateGGXPureColor(Color::WhiteDarken(0.25f), 0.05f, 0.0f, 0.5f), // Color::WhiteDarken(0.25f)
        float3{0.0f, 0.0f, 0.0f},
        EObjectType::OBJ_CUBE
    );
    */
    //std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f}, color::Red(), color::Green(), color::YellowLighten(0.9f) * 50.0f, 0.5f); // 
    
    std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f}, color::White(), color::White(), color::YellowLighten(0.98f) * 50.0f, 0.5f); // 
    //std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({ 0.0f, 0.0f,0.0f }, color().White(), color().White(), color().YellowLighten(0.98f) * 50.0f, 0.5f);
    
    //std::vector<SceneObject> CornellBox1 = SceneHelper::CreateDiffuseCornellBox({0.0f, 0.0f,0.0f}, Color::Red(), Color::Green(), Color::YellowLighten(0.9f) * 50.0f, 0.5f); // 
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTestScene({0.0f, 0.0f,0.0f}, 1.0f, 20.0f, 0.5f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateSpectrumTestScene({0.0f, 0.0f,0.0f});
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateBigGlassTestScene({0.0f, 0.0f,0.0f}, 1.0);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateCubeTestScene({0.0f, 0.0f,0.0f}, 3.0f, 5.0f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateAbbeTestScene({0.0f, 0.0f,0.0f}, 5.0f, 2.0f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTestScene2({0.0f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Green(), EObjectType::OBJ_CUBE);
    std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTorus({0.0f, 0.0f,0.0f}, 1.25f, 1.0f, 2);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassSDFVolume(0, {0.0f, 0.0f,0.0f}, 1.1f, 2.0f, 1);
    Scene.insert(Scene.end(), CornellBox1.begin(), CornellBox1.end());
    Scene.insert(Scene.end(), GlassScene.begin(), GlassScene.end());
    //
    /*
    std::vector<SceneObject> CornellBox2 = SceneHelper::CreateCornellBox({1.02f, 0.0f,0.0f},
        Color::Pink(),
        Color::Cyan(),
        Color::YellowLighten(0.9f) * 50.0f);
    std::vector<SceneObject> GlassMatrix = SceneHelper::CreateGlassTestScene2({1.02f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Red(), EObjectType::OBJ_SPHERE);
    Scene.insert(Scene.end(), CornellBox2.begin(), CornellBox2.end());
    Scene.insert(Scene.end(), GlassMatrix.begin(), GlassMatrix.end());
    //
    std::vector<SceneObject> CornellBox3 = SceneHelper::CreateCornellBox({-1.02f, 0.0f,0.0f},
        Color::Pink(),
        Color::Cyan(),
        Color::YellowLighten(0.9f) * 50.0f);
    std::vector<SceneObject> GlassMatrix2 = SceneHelper::CreateGlassTestScene2({-1.02f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Green(), EObjectType::OBJ_CUBE);
    Scene.insert(Scene.end(), CornellBox3.begin(), CornellBox3.end());
    Scene.insert(Scene.end(), GlassMatrix2.begin(), GlassMatrix2.end());
    //
    std::vector<SceneObject> CornellBox4 = SceneHelper::CreateCornellBox({0.0f, 1.04f,0.0f},
        Color::Purple(),
        Color::Yellow(),
        Color::YellowLighten(0.9f) * 150.0f,
        0.25);
    std::vector<SceneObject> GlassMatrix3 = SceneHelper::CreateGlassTestScene2({0.0f, 1.04f,0.0f}, 1.25f, 1.5f, 2, Color::Cyan(), EObjectType::OBJ_SPHERE, 0.25f);
    Scene.insert(Scene.end(), CornellBox4.begin(), CornellBox4.end());
    Scene.insert(Scene.end(), GlassMatrix3.begin(), GlassMatrix3.end());
    */
    return Scene;
}

int main()
{
    //spectrum::query::SelfTestSimple();
    //spectrum::sample::ValidationSampleCIEXYZ();
    //spectrum::ValidationSpectrum();
    
    srand(static_cast<unsigned>(time(NULL)));



    //如果想用回原来的玻璃圆环或各种矩阵测试场景，取消注释下面这行：
    //const std::vector<SceneObject> scene = CreateScene();



    // 目前启用的是 Blender 外部 SDF 加载场景：
    std::vector<SceneObject> scene;
    std::vector<SceneObject> cornellBox = SceneHelper::CreateCornellBox(
        float3{ 0.0f, 0.0f, 0.0f }, color::Red(), color::Green(), color::YellowLighten(0.98f) * 50.0f, 0.5f
        //float3{ 0.0f, 0.0f, 0.0f }, color::White(), color::White(), color::YellowLighten(0.98f) * 50.0f, 0.5f
    );
    scene.insert(scene.end(), cornellBox.begin(), cornellBox.end());

    // 读取硬盘上的 .bin 数据
    const int SDF_RESOLUTION = 128;
    const std::string SDF_PATH = "D:/blender_sdf_128.bin";
    float* sdfVolumeGPU = SceneHelper::LoadSDFVolumeToGPU(SDF_PATH, SDF_RESOLUTION);

    // SDF_ID 分配 (我们固定用 0)
    AdditionalObjectInfo customSDFInfo = sdf::CreateSDFVolume(0);

    if (sdfVolumeGPU) {
        scene.emplace_back(
            float3{ 0.0f, 0.0f, 0.0f },
            float3{ 0.5f, 0.5f, 0.5f },
            Material::CreateGlassPureColor(
                color::White(), // 已修正为小写 color
                1.5f,           // 2. IOR (折射率)：1.5 是水晶和高级玻璃的真实物理折射率
                5.0f,           // 3. 内部散射/衰减参数 (保持 5.0 即可)
                12.0f           // 4. 阿贝数 (Abbe)：决定七彩光的强弱！越低色散越强，12.0 能产生极其绚丽的彩虹光斑🌈
            ),
            float3{ 0.0f, 0.0f, 0.0f },
            EObjectType::OBJ_SDF,
            customSDFInfo
        );
    }
    else {
        scene.emplace_back(
            float3{ 0.0f, 0.0f, 0.0f },
            float3{ 0.3f, 0.3f, 0.3f },
            Material().CreateGlassPureColor(color::White(), 1.5f, 15.0f, 0.05f), // 🔴 已修正为小写 color
            float3{ 0.0f, 0.0f, 0.0f },
            EObjectType::OBJ_SPHERE
        );
    }

    std::vector<SceneObject> sceneBase = scene;

    cudaTextureObject_t texAlbedo = LoadCUDATexture("D:/c++/ApertureRendererV3-OptiX-Geometry/tex/Retopo_node_0_Bake1_PBR StoA_Diffuse.png");
    cudaTextureObject_t texNormal = LoadCUDATexture("D:/c++/ApertureRendererV3-OptiX-Geometry/tex/Retopo_node_0_Bake1_PBR StoA_Normal.png");
    cudaTextureObject_t texMetallic = LoadCUDATexture("D:/c++/ApertureRendererV3-OptiX-Geometry/tex/Retopo_node_0_Bake1_PBR StoA_Metalness.png");
    cudaTextureObject_t texRoughness = LoadCUDATexture("D:/c++/ApertureRendererV3-OptiX-Geometry/tex/Retopo_node_0_Bake1_PBR StoA_Roughness.png");

    // 2. 读取 OBJ 文件并把数千个三角形推入 scene 数组中
    SceneHelper::LoadTexturedMesh(
        99,                             // 物体ID
        "D:/c++/ApertureRendererV3-OptiX-Geometry/obj/dragon.obj",           // 注意替换为你本地真实的 OBJ 路径
        make_float3(0.0f, -0.2f, 0.0f), // 模型摆放的世界坐标 (Y 轴微调防止穿模)
        1.0f,                           // 缩放比例
        texAlbedo,                     // 颜色贴图
        texNormal,//法线贴图
        texMetallic,//金属度
        texRoughness,//粗糙度
        scene                           // 传入你的场景大数组
    );


    SceneRenderer sceneRendererBase(sceneBase); // 浅 BVH 树，速度极快 (15帧)
    SceneRenderer sceneRendererGun(scene);

    // 显存缓存注册区 (SDF 八叉树加速结构)，注册外部导入的 Blender 体积缓存 (仅当加载成功时注册)
    if (sdfVolumeGPU) {
        sdf::SDFVolumeInfo volumeInfo;
        volumeInfo.volumePtr_ = sdfVolumeGPU;
        volumeInfo.dim_ = make_int3(SDF_RESOLUTION, SDF_RESOLUTION, SDF_RESOLUTION);
        volumeInfo.ComputeVoxelSize();

        /*sceneRenderer.AddVolumeToCacheManager(
            { 64, 64, 64 },
            { 0.0f, 0.0f, 0.0f },
            { 1.0f, 1.0f, 1.0f },
            sdf::SDFVolumeFunctor(volumeInfo),
            customSDFInfo.sdfInfo_
        );*/
        sceneRendererBase.AddVolumeToCacheManager({ 64, 64, 64 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, sdf::SDFVolumeFunctor(volumeInfo), customSDFInfo.sdfInfo_);
        sceneRendererGun.AddVolumeToCacheManager({ 64, 64, 64 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, sdf::SDFVolumeFunctor(volumeInfo), customSDFInfo.sdfInfo_);
        std::cout << "Custom SDF Volume registered to Cache Manager." << std::endl;
    }

    {
        const AdditionalObjectInfo diamondSDF = sdf::CreateDiamond();
        /*sceneRenderer.AddVolumeToCacheManager(
            {64, 64, 64},
            {0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f},
            sdf::SDFDiamondFunctor{},
            diamondSDF.sdfInfo_
        );*/
        sceneRendererBase.AddVolumeToCacheManager({ 64, 64, 64 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, sdf::SDFDiamondFunctor{}, diamondSDF.sdfInfo_);
        sceneRendererGun.AddVolumeToCacheManager({ 64, 64, 64 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, sdf::SDFDiamondFunctor{}, diamondSDF.sdfInfo_);
    }

    const float3 cameraPosition = normalize(float3{ 0.0f,0.0f,0.8f }) * 1.1f;
     
    std::cout << "fill volume done." << std::endl;

    Camera camera(sceneRendererGun, "test camera", 1280 / 2, 720 / 2);
    camera.SetPosition(cameraPosition);

    std::cout << "prepare rendering." << std::endl;

    //camera.RenderToFile("output_beautiful", 4096, ACES, 1.0f);

    RunGUI(camera, &sceneRendererBase, &sceneRendererGun);
    
    return 0;
}