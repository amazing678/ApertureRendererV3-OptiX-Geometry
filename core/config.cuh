// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#define INTERSECT_EPS(depth) (1.0e-3f * expf(fminf(static_cast<float>(depth) * 0.5f, 4.0f)))
#define DIV_ZERO_EPS 1.0e-12

#define MAX_DELTA_SURFACE_DEPTH 1024
#define MIN_TRACING_DEPTH 3
#define MAX_TRACING_DEPTH 1000
#define PERFECT_DELTA_SURFACE 0.99f

#define DELTA_RR_TERMINATE_REDUCE 0.99f
#define RADIANCE_CLAMP 1.0e3f

#define NAN_REPLACEMENT 0.0f
#define INF_REPLACEMENT 0.0f

#define BVH_USE_16_BITS_NODE
#define BVH_PACK_NODE
#define BVH_USE_SORT_STACK

#if defined(BVH_PACK_NODE) && (!defined(BVH_USE_16_BITS_NODE))
#  error "BVH_PACK_NODE requires BVH_USE_16_BITS_NODE=1. Define BVH_USE_16_BITS_NODE=1 or disable BVH_PACK_NODE."
#endif

#define NEE_MAX_NON_DELTA_DEPTH 10
#define NEE_USE_CUSTOM_FACET_WEIGHT

#define POWER_HEURISTIC

//#define PATH_GUIDING_USE_NEE
#define PATH_GUIDING_COLLECT_MAX_BRIGHTNESS 1.0e5f
#define PATH_GUIDING_COLLECT_MIN_BRIGHTNESS 1.0e-5f
#define PATH_GUIDING_PROBE_MIN_BRIGHTNESS 1.0e-5f
#define PATH_GUIDING_PROBE_MAX_STORE_COUNT 1000000ull // forget

#define PATH_GUIDING_MAX_ALPHA 0.5
#define PATH_GUIDING_LERP_MAX_FRAME 64
#define PATH_GUIDING_LERP_POW 0.5

#define PATH_GUIDING_RESOLUTION_LEVEL 4

#define PATH_GUIDING_COLLECT_DEPTH 5 // TODO: collect NEE samples too
#define PATH_GUIDING_RESOLUTION_UNIT 0.05
//#define PATH_GUIDING_RESOLUTION_UNIT 0.2

#define PATH_GUIDING_REFLECT_VECTOR
#define PATH_GUIDING_CLARBERG
#define PATH_GUIDING_ICOSAHEDRON

//#define DEBUG_PATH_GUIDING_INDIRECT_VOLUME
#define DEBUG_PATH_GUIDING_INDIRECT_VOLUME_DETAILED
#define DEBUG_PATH_GUIDING_INDIRECT_VOLUME_CULL_INVALID_DIRECTION

//#define DEBUG_FORCE_ALL_PURE_DIFFUSE
//#define DISABLE_SKYLIGHT

// spectrum options
#define USE_PATH_GUIDING // for temporally debug
#define USE_3_CHANNEL_PROBE
#define PROBES_QUERY_EPS 1.0e-6
#define USE_SPECTRUM_RENDERING

#define SPECTRUM_RGB_LUT_RES 512 // 512^3 x float4 -> 2 GiB, 256^3 x float4 -> 0.5 GiB
#define SPECTRUM_LAMBDA_LUT_RES 840 // 840 x float4, 420

constexpr size_t SIZE_T_MAX = ~static_cast<size_t>(0);
#ifdef PATH_GUIDING_ICOSAHEDRON
#define PATH_GUIDING_FACE_COUNT 20
#else
#define PATH_GUIDING_FACE_COUNT 6
#endif

// SDF
#define SDF_EPS(depth) (INTERSECT_EPS(depth) * 0.5f)
#define SDF_STEPS 512
#define SDF_VOLUME_STEP_SCALE 0.9f
#define SDF_VOLUME_USE_NEAREST_SCALE 2.5f
#define SDF_UNROLL

#if defined(__CUDA_ARCH__) && defined(SDF_UNROLL)
#define SDF_CUDA_UNROLL #pragma unroll
#else
#define SDF_CUDA_UNROLL {}
#endif
