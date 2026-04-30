cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Aperture LANGUAGES CXX CUDA)

enable_language(CUDA)

# --- 基础配置 ---
option(CUDA_SANITIZER OFF)
option(GUI ON)
option(RTX30XX OFF)
option(TENSORCORE OFF)

if (WIN32)
    message(STATUS "Current Platform: Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /Ot")
    # 忽略 Release 模式下常见的库冲突警告
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    add_compile_definitions($<$<CXX_COMPILER_ID:MSVC>:NOMINMAX _CRT_SECURE_NO_WARNINGS>)
else ()
    message(STATUS "Current Platform: Linux")
    add_definitions(-DLINUX)
endif ()

if (TENSORCORE)
    add_definitions(-DENBALE_TENSOR)
endif()

# --- 搜集核心源代码 ---
file(GLOB_RECURSE CORE_HEADERS
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cuh")
file(GLOB_RECURSE CORE_SOURCES
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu")

source_group("Core Include" FILES ${CORE_HEADERS}) 
source_group("Core Source" FILES ${CORE_SOURCES})
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/core/render/optix_device.cu")#optix着色器不参与主程序编译
# =========================================================
# 1. 定义可执行文件目标 (必须在 target_link_libraries 之前)
# =========================================================
add_executable(ApertureRenderer ${CORE_HEADERS} ${CORE_SOURCES} main.cu)

# =========================================================
# 2. 设置目标属性与编译选项
# =========================================================
set_target_properties(ApertureRenderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
)

target_compile_definitions(ApertureRenderer PRIVATE _HAS_STD_BYTE=0)

# 包含核心头文件
target_include_directories(ApertureRenderer PRIVATE
    ${PROJECT_SOURCE_DIR}/core
    "D:/NvidiaOptiX/include"
    "D:/c++/glew-2.3.0/include"            
    "D:/c++/glfw-3.4.bin.WIN64/include"
)

# CUDA 架构设置
if (RTX30XX)
    target_compile_options(ApertureRenderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168 --extended-lambda>)
else()
    target_compile_options(ApertureRenderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168 --extended-lambda>)
endif()

# =========================================================
# 3. GUI 与 第三方库链接逻辑 (在目标定义之后)
# =========================================================
if (GUI)
    add_definitions(-DGUI)

    # --- GLEW 逻辑 ---
    find_package(GLEW QUIET)
    if(NOT TARGET GLEW::GLEW)
        message(STATUS "GLEW not found by find_package, using manual paths...")
        add_library(GLEW::GLEW UNKNOWN IMPORTED)
        set_target_properties(GLEW::GLEW PROPERTIES
            IMPORTED_LOCATION "${GLEW_LIBRARY_RELEASE}"
            IMPORTED_LOCATION_DEBUG "${GLEW_LIBRARY_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "D:/c++/glew-2.3.0/include"
        )
    endif()

    # --- GLFW 逻辑 ---
    find_package(glfw3 QUIET)
    if(NOT TARGET glfw)
        message(STATUS "GLFW not found by find_package, using manual paths...")
        add_library(glfw UNKNOWN IMPORTED)
        set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${GLFW_LIBRARY_RELEASE}"
            IMPORTED_LOCATION_DEBUG "${GLFW_LIBRARY_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "D:/c++/glfw-3.4.bin.WIN64/include"
        )
    endif()

    # 添加 Imgui 子项目
    add_subdirectory(Imgui)
    
    # --- 核心链接步骤 ---
    # 加上 opengl32 是为了解决那 11 个无法解析的外部符号错误
    target_link_libraries(ApertureRenderer PRIVATE 
        GLEW::GLEW 
        glfw 
        Imgui 
        opengl32
    )
    
    add_dependencies(ApertureRenderer Imgui)
endif()

# --- CUDA Sanitizer 调试工具配置 (保持原样) ---
if (CUDA_SANITIZER)
    target_compile_options(ApertureRenderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -G -O0>)
    target_compile_definitions(ApertureRenderer PRIVATE CUDA_SANITIZER_ENABLED=1)
endif()

set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# 源文件和目标 PTX 文件路径
set(OPTIX_CU_FILE "${PROJECT_SOURCE_DIR}/core/render/optix_device.cu")
set(OPTIX_PTX_FILE "${PTX_OUTPUT_DIR}/optix_device.ptx")

# 动态判断架构 (跟你上面的一致)
if (RTX30XX)
    set(PTX_ARCH "sm_86")
else()
    set(PTX_ARCH "sm_75")
endif()

# 自定义编译命令：调用 nvcc 把 .cu 转成 .ptx
add_custom_command(
    OUTPUT ${OPTIX_PTX_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -O3 -arch=${PTX_ARCH} --use_fast_math
            -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH   #强制跳过微软的版本检查！
            -D_HAS_STD_BYTE=0
            -I"${PROJECT_SOURCE_DIR}/core" 
            -I"D:/NvidiaOptiX/include" 
            ${OPTIX_CU_FILE} -o ${OPTIX_PTX_FILE}
    DEPENDS ${OPTIX_CU_FILE}
    COMMENT "正在编译 OptiX 微型着色器到 PTX..."
)

# 创建一个自定义目标，并让主程序依赖它，保证每次编译前 PTX 都是最新的
add_custom_target(OptixPTXTarget ALL DEPENDS ${OPTIX_PTX_FILE})
add_dependencies(ApertureRenderer OptixPTXTarget)