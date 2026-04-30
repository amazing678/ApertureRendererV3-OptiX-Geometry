## Install

### Requirements
- CUDA 11.x
  |Version|Compatible|Info|
  |-|-|-|
  |11.0|Yes*|CUDA11.0 Only support sm_75|
  |11.1|Yes|Recommended|
  |11.2|Yes|Recommended|
  |11.4~11.7|Not Sure||
  |11.8|Yes*|
- Glfw3 and GLEW if you want to compile with GUI, or use **vcpkg**
- x64 Ubuntu or Windows
- Nvidia GPU that supports sm_75 or sm_86

### To build on Linux:
- Install CUDA 11.x.
- Run CMake to generate the project, and use `-T cuda=<PATH/TO/CUDA/toolkit>` to select a cuda version.
- If you are using the RTX3000 series (or another sm_86 card), enable the `RTX30XX` option.
- Compile your project, it may stuck for a while.
- Linux currently doesn't support build with GUI.

### To build on Windows:
- Install CUDA 11.x.
- Install glew and glfw3 if you need GUI support, or use **vcpkg**. (Remember to install the x64 version)
- Run CMake to generate the project, and use `-T cuda=<PATH/TO/CUDA/toolkit>` to select a cuda version.
- If you are using CUDA 11.0~11.3, you may also need to have MSVC v14.25 and using `-T version=14.25`.
- If you are using the RTX3000 series, enable the `RTX30XX` option.
- Check the `GUI` option if needed.
- Compile your project, it may stuck for a while.