// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif


#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "nee/nee.cuh"
#include "guiding/octahedron.cuh"
#include "vector.cuh"
#include "sample.cuh"
#include "object.cuh"
#include "material.cuh"
#include "intersection.cuh"
#include "denoise/denoise_statistic.cuh"
#include "guiding/guiding_debug.cuh"
#include "guiding/guiding.cuh"
#include "spectrum/sample_ciexyz.cuh"

#ifndef RENDER_GLOBAL
#define RENDER_GLOBAL
__device__ inline __constant__ int RESOLUTION;
__device__ inline __constant__ int FRAME_INDEX = 0;

//
__device__ inline __constant__ SceneObject* SCENE_OBJECTS;
__device__ inline __constant__ int SCENE_OBJECT_COUNTS = 0;

__device__ inline __constant__ AdditionalLightInfo* SCENE_LIGHTS_INFO;
__device__ inline __constant__ int SCENE_LIGHTS_COUNTS = 0;

//
__device__ inline __constant__ BVHNode* BVH_NODES;
__device__ inline __constant__ int BVH_NODES_COUNTS = 0;

__device__ inline __constant__ int* BVH_OBJECT_INDICES = 0;

// settings
__device__ inline __constant__ SceneSetting SETTING;

// for path guiding
__device__ inline __constant__ Octahedron* SCENE_PROBES;
__device__ inline __constant__ size_t SCENE_PROBES_COUNTS = 0;

__device__ inline __constant__ int3 INDIRECT_VOLUME_START = {0,0,0};
__device__ inline __constant__ int3 INDIRECT_VOLUME_END = {0,0,0};
__device__ inline __constant__ int3 INDIRECT_VOLUME_SIZE = {0,0,0};
__device__ inline __constant__ size_t* INDIRECT_INDEX_VOLUME;

__device__ inline __constant__ float* SPECTRUM_LUT_RGB;
__device__ inline __constant__ float* SPECTRUM_LUT_LAMBDA;
#endif

[[nodiscard]] FUNCTION_MODIFIER_DEVICE_INLINE float3 SkyLighting(float3 rayDirection)
{
    [[maybe_unused]] const float3 skyWhite = float3{1.0f, 1.0f, 1.0f};
    [[maybe_unused]] const float3 skyBlue = float3{0.05f, 0.45f, 1.0f};
    [[maybe_unused]] const float3 skyDown = float3{0.2f, 0.6f, 1.0f} * 0.125f;
    [[maybe_unused]] const float3 emissive = 0.5f * lerp(
        lerp(skyBlue, skyWhite, powf(fmaxf(rayDirection.y, 0.0f), 0.75f)),
        skyDown,
        fminf(fmaxf(-rayDirection.y * 0.5f + 0.5f, 0.0f), 1.0f)
    );
#ifdef DISABLE_SKYLIGHT
    return {0.0f, 0.0f, 0.0f};
#else
    return emissive;
#endif
}

template<bool bUseBVH, bool offsetHitPosition = true>
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_DEVICE_INLINE TraceContext Trace(
    const int depth,
    const float lambda,
    const SpectrumBasis& lambdaBasis,
#else
FUNCTION_MODIFIER_DEVICE_INLINE TraceContext Trace(
#endif
    curandState* seed, float3 rayOrigin, float3 rayDirection, const MediaContext& currentMediaContext)
{
    TraceContext traceResult;
    const intersection::SceneObjectsIntersector intersector{SCENE_OBJECTS, SCENE_OBJECT_COUNTS};
    const float currentEPS = INTERSECT_EPS(depth);
    if(currentMediaContext.bInsideMedia_ && currentMediaContext.objectIndex_ >= 0)
    {
        traceResult.intersectionContext_ = Intersect(depth, rayOrigin, rayDirection, intersector, currentMediaContext.objectIndex_, currentEPS); // solo trace, assume object no overlap
    }
    else
    {
        if constexpr (bUseBVH)
        {
            traceResult.intersectionContext_ = intersection::TraverseBVH<false>(depth, rayOrigin, rayDirection, intersector,
                    BVH_NODES,
                    BVH_NODES_COUNTS,
                    BVH_OBJECT_INDICES,
                    INTERSECT_EPS(depth)
                );
        }
        else
        {
            traceResult.intersectionContext_ = Intersect(depth, rayOrigin, rayDirection, intersector, -1, currentEPS);
        }
    }
    if(traceResult.intersectionContext_.bHit_ == false && currentMediaContext.bInsideMedia_) // should out
    {
        traceResult.sampleContext_.bMissThinSurfaceInsideMedia_ = true; // inside media, but not hit, is not possible
#ifdef USE_SPECTRUM_RENDERING
        traceResult.sampleContext_.hitBRDF_ = 1.0f;
#else
        traceResult.sampleContext_.hitBRDF_ = {1.0f, 1.0f, 1.0f};
#endif
        traceResult.sampleContext_.nextDirection_ = rayDirection;
        traceResult.sampleContext_.nextDirectionPDF_ = 1.0f;
        traceResult.intersectionContext_.hitLocalPosition_ = rayOrigin;
        //traceResult.intersectionContext_.hitNormal_ = ?
        traceResult.intersectionContext_.objectIndex_ = currentMediaContext.objectIndex_;
        traceResult.sampleContext_.bDeltaSurface_ = true;
        traceResult.sampleContext_.hitEmissive_ = {0.0f, 0.0f, 0.0f};
        traceResult.sampleContext_.mediaContext_.bInsideMedia_ = false;
        traceResult.sampleContext_.mediaContext_.vertexLocation_ =  traceResult.intersectionContext_.hitLocalPosition_;
    }
    if(traceResult.intersectionContext_.bHit_)
    {
        const SceneObject& currentSceneObject = SCENE_OBJECTS[traceResult.intersectionContext_.objectIndex_];
        traceResult.shadingModel_ = currentSceneObject.material_.shadingModel_;
#ifdef USE_SPECTRUM_RENDERING
        traceResult.sampleContext_ = EvaluateMaterial(
            lambda, lambdaBasis, SPECTRUM_LUT_RGB,
#else
        traceResult.sampleContext_ = EvaluateMaterial(
#endif
            seed,
            traceResult.shadingModel_,
            rayDirection,
            traceResult.intersectionContext_,
            currentMediaContext);
    }
    else
    {
#ifdef USE_SPECTRUM_RENDERING
        traceResult.sampleContext_.hitBRDF_ = 1.0f;
#else
        traceResult.sampleContext_.hitBRDF_ = {1.0f, 1.0f, 1.0f};
#endif
        traceResult.sampleContext_.bDeltaSurface_ = false;
        traceResult.sampleContext_.nextDirection_ = rayDirection;
        traceResult.sampleContext_.nextDirectionPDF_ = 1.0f;
        traceResult.sampleContext_.hitEmissive_ = SkyLighting(rayDirection);  // or get sky lighting
        traceResult.sampleContext_.mediaContext_.bInsideMedia_ = false;
    }
    if constexpr (offsetHitPosition)
    {
        const float offsetEPS = currentEPS * fmaxf(1.0f, length(traceResult.intersectionContext_.hitPosition_)); // TODO: if convex, no offset needed
        traceResult.intersectionContext_.hitPosition_ = traceResult.intersectionContext_.hitPosition_ + sign(dot(traceResult.intersectionContext_.hitNormal_, traceResult.sampleContext_.nextDirection_)) * traceResult.intersectionContext_.hitNormal_ * offsetEPS * (Rand1(seed) + 0.5f);
        //traceResult.intersectionContext_.hitPosition_ = traceResult.intersectionContext_.hitPosition_ + traceResult.sampleContext_.nextDirection_ * offsetEPS;
    }
    return traceResult;
}

// return true if visible
// sourcePoint better add eps first
template<bool bUseBVH, typename Intersector = intersection::SceneObjectsIntersector>
FUNCTION_MODIFIER_DEVICE_INLINE IntersectionContext SegmentTrace(const int depth, const float3& sourcePoint, const float3& targetPoint, const Intersector& intersector)
{
    if constexpr (bUseBVH)
    {
        return AnyHitSegmentBVH(depth,
                sourcePoint, targetPoint, intersector,
                BVH_NODES,
                BVH_NODES_COUNTS,
                BVH_OBJECT_INDICES
            );
    }
    else
    {
        return AnyHitSegment(depth, sourcePoint, targetPoint, intersector);
    }
}

FUNCTION_MODIFIER_INLINE float PowerHeuristic(const float pA, const float pB)
{
#ifdef POWER_HEURISTIC
    const float a = pA * pA;
    const float b = pB * pB;
#else
    const float a = pA;
    const float b = pB;
#endif
    const float misW = a / fmaxf(1e-24f, a + b);
    return misW;
}

FUNCTION_MODIFIER_DEVICE_INLINE float GetPathGuidingProb(const GBuffer& gbuffer)
{
    const float usePathGuidingProb = lerp(0.0f, PATH_GUIDING_MAX_ALPHA, powf(fminf(1.0f, static_cast<float>(FRAME_INDEX) / PATH_GUIDING_LERP_MAX_FRAME), PATH_GUIDING_LERP_POW));
    //const float usePathGuidingProb = PATH_GUIDING_MAX_ALPHA;//surfaceAlpha;//lerp(0.0f, PATH_GUIDING_MAX_ALPHA, powf(fminf(1.0f, static_cast<float>(FRAME_INDEX) / PATH_GUIDING_LERP_MAX_FRAME), PATH_GUIDING_LERP_POW));
    return saturate_(usePathGuidingProb);
}

// rayDirection towards hit surface
template<bool bUseBVH, bool bUsePathGuiding>
#ifdef USE_SPECTRUM_RENDERING
FUNCTION_MODIFIER_DEVICE float NEE(
    const int depth,
    const float lambda,
    const SpectrumBasis& lambdaBasis,
#else
FUNCTION_MODIFIER_DEVICE float3 NEE(
#endif
    curandState* seed,
#ifdef USE_3_CHANNEL_PROBE
    const float3 maskForProbeQuery,
#endif
    const EShadingModel shadingModel,
    const float3 rayDirection,
    const IntersectionContext& currentIntersection,
    const Octahedron* pathGuidingProbe,
    bool* bIsValidPtr,
    float3* lightDirectionPtr,
#ifdef USE_SPECTRUM_RENDERING
    float* outFCos,
    float* outLe,
#else
    float3* outFCos,
    float3* outLe,
#endif
    float* outPLightOmega
    )
{
    *bIsValidPtr = false;
    const LightSample NEELightSample = NEESampleLight(SCENE_LIGHTS_INFO, SCENE_OBJECTS, SCENE_LIGHTS_COUNTS, currentIntersection.hitPosition_, seed);
    if (NEELightSample.bIsValid_ == false)
    {
        *bIsValidPtr = false;
#ifdef USE_SPECTRUM_RENDERING
        return 0.0f;
#else
        return {0.0f, 0.0f, 0.0f};
#endif
    }
    const float pATotal = fmaxf(1e-20f, NEELightSample.pA_ * NEELightSample.pFacet_);
    //
    const float3 lightDirectionUnnormalized  = NEELightSample.samplePoint_ - currentIntersection.hitPosition_;
    const float distance2_NEE = fmaxf(dot(lightDirectionUnnormalized, lightDirectionUnnormalized), DIV_ZERO_EPS);
    const float3 lightDirection = lightDirectionUnnormalized * rsqrtf(distance2_NEE);
    *lightDirectionPtr = lightDirection;
    //
    const intersection::SceneObjectsIntersector intersector{SCENE_OBJECTS, SCENE_OBJECT_COUNTS};
    const IntersectionContext segmentTraceResult = SegmentTrace<bUseBVH>(depth, currentIntersection.hitPosition_, NEELightSample.samplePoint_, intersector); // actually shadow ray
    if (segmentTraceResult.bHit_)
    {
        *bIsValidPtr = true; // TODO: true but occluded
#ifdef USE_SPECTRUM_RENDERING
        return 0.0f;
#else
        return {0.0f, 0.0f, 0.0f};
#endif
    }
    const float cosX_NEE= fmaxf(0.0f, dot(currentIntersection.hitNormal_, lightDirection));
    const float cosY_NEE= fmaxf(0.0f, dot(NEELightSample.samplePointNormal_, -lightDirection));
    //
    if (cosX_NEE <= 0.0f || cosY_NEE <= 0.0f)
    {
        *bIsValidPtr = false; // treat as shadowed
#ifdef USE_SPECTRUM_RENDERING
        return 0.0f;
#else
        return {0.0f, 0.0f, 0.0f};
#endif
    }
#ifdef USE_SPECTRUM_RENDERING
    const float BRDFCosNEE = EvaluateBSDFWithCos(
        lambda, lambdaBasis, SPECTRUM_LUT_RGB,
#else
    const float3 BRDFCosNEE = EvaluateBSDFWithCos(
#endif
        shadingModel,
        rayDirection,
        lightDirection,
        currentIntersection,
        {} // should be media state, but we SKIP NEE WHEN INSIDE MEDIA
    );
    const float G = (cosY_NEE / distance2_NEE);
#ifdef USE_SPECTRUM_RENDERING
    const float lambdaResponsiveness = spectrum::lut::Interact(lambdaBasis, NEELightSample.sampleEmissive_, SPECTRUM_LUT_RGB);
    const float contributionNEE = BRDFCosNEE * lambdaResponsiveness * (G / pATotal); // = fs * Le * G / pA_total
#else
    const float3 contributionNEE = BRDFCosNEE * NEELightSample.sampleEmissive_ * (G / pATotal); // = fs * Le * G / pA_total
#endif
    *bIsValidPtr = true;
    // MIS in solid angle
    const float pLightOmega = NEELightPDFOmegaFromArea(currentIntersection.hitPosition_, NEELightSample);
    // record for PathGuiding
    *outFCos = BRDFCosNEE;
#ifdef USE_SPECTRUM_RENDERING
    *outLe = lambdaResponsiveness;
#else
    *outLe = NEELightSample.sampleEmissive_;
#endif
    *outPLightOmega = pLightOmega;
    //
    float pBsdfOmega = fmaxf(
#ifdef USE_SPECTRUM_RENDERING
        BSDF_PDF(
            lambdaBasis, SPECTRUM_LUT_RGB,
#else
        BSDF_PDF(
#endif
            shadingModel,
            currentIntersection.gbuffer_,
            currentIntersection.hitPosition_,
            currentIntersection.hitNormal_,
            rayDirection,
            lightDirection), 0.0f);
    if constexpr (bUsePathGuiding)
    {
        if(pathGuidingProbe != nullptr)
        {
#ifdef USE_SPECTRUM_RENDERING
#ifdef USE_3_CHANNEL_PROBE
            const float pProbeAtNEEDirection = pathGuidingProbe->OctahedronHemispherePDF(lightDirection, currentIntersection.hitNormal_, maskForProbeQuery);
#else
            const float pProbeAtNEEDirection = pathGuidingProbe->OctahedronHemispherePDF(lightDirection, currentIntersection.hitNormal_);
#endif
#else
#ifdef USE_3_CHANNEL_PROBE
            const float pProbeAtNEEDirection = pathGuidingProbe->OctahedronHemispherePDF(lightDirection, currentIntersection.hitNormal_, maskForProbeQuery);
#else
            const float pProbeAtNEEDirection = pathGuidingProbe->OctahedronHemispherePDF(lightDirection, currentIntersection.hitNormal_);
#endif
#endif
            const float usePathGuidingProb = GetPathGuidingProb(currentIntersection.gbuffer_);
            pBsdfOmega = (1.0f - usePathGuidingProb) * pBsdfOmega + usePathGuidingProb * pProbeAtNEEDirection;
        }
    }
    const float misW = PowerHeuristic(pLightOmega, pBsdfOmega);
    return contributionNEE * misW;
}

struct NEEState
{
    bool previousDidNEE_ = false;
    
    float lastSegmentBSDFPDF_ = 0.0f;
    float3 lastVertexPosition_ = {0.0f, 0.0f, 0.0f};
};

FUNCTION_MODIFIER_DEVICE_INLINE void CollectRadiance(
    float3* radianceSampleDirections,
#ifdef USE_SPECTRUM_RENDERING
    float* radianceSamples,
    float* radianceMasks,
#else
    float3* radianceSamples,
    float3* radianceMasks,
#endif
    float3* radianceVertexLocation,
    float3* radianceVertexDirection,
    bool* radianceValid,
#ifdef USE_SPECTRUM_RENDERING
    const float endpointLe,
#else
    const float3 endpointLe,
#endif
    const float3 nextDirection,
#ifdef USE_SPECTRUM_RENDERING
    const float segmentContrib,
#else
    const float3 segmentContrib,
#endif
    const float rrSurviveProb,
    const int nonDeltaDepth,
    const bool allowStartAtThisVertex,
    const bool advanceMasks,
    [[maybe_unused]] const float3 currentRayDirection,
    const float3 currentHitLocation,
    const float3 currentHitNormal,
    const int maxDepth
)
{
    if (!allZero(endpointLe))
    {
        const int hiAcc = min(nonDeltaDepth, maxDepth);
        for (int i = 0; i < hiAcc; ++i)
        {
            if (!allZero(radianceMasks[i]))
            {
                radianceSamples[i] = radianceSamples[i] + radianceMasks[i] * endpointLe;
                radianceValid[i] = true;
            }
        }
    }

    // record mask
    if (allowStartAtThisVertex && nonDeltaDepth < maxDepth)
    {
        if (allZero(radianceMasks[nonDeltaDepth]))
        {
            radianceSampleDirections[nonDeltaDepth] = nextDirection;
            radianceVertexLocation[nonDeltaDepth] = currentHitLocation;
#ifdef PATH_GUIDING_REFLECT_VECTOR
            // currentRayDirection points toward surface, reflect(currentRayDirection, currentHitNormal) points outward surface
            radianceVertexDirection[nonDeltaDepth] = reflect(currentRayDirection, currentHitNormal);
#else
            radianceVertexDirection[nonDeltaDepth] = currentHitNormal;
#endif
#ifdef USE_SPECTRUM_RENDERING
            radianceMasks[nonDeltaDepth] = 1.0f;
#else
            radianceMasks[nonDeltaDepth] = float3{1.0f, 1.0f, 1.0f};
#endif
        }
    }
    
    if (advanceMasks)
    {
        const float invQ = 1.0 / (rrSurviveProb);
        const int hiUpd = min(nonDeltaDepth + (allowStartAtThisVertex ? 1 : 0), maxDepth);
        for (int i = 0; i < hiUpd; ++i)
        {
            if (!allZero(radianceMasks[i]))
            {
                radianceMasks[i] = radianceMasks[i] * (segmentContrib * invQ);
            }
        }
    }
}

FUNCTION_MODIFIER_DEVICE_INLINE Octahedron* QueryProbe(const float3 position, const float3 direction)
{
    const int4 voxelGridWS = pg::GetVoxelGrid(position, direction);
    const int lx = voxelGridWS.x - INDIRECT_VOLUME_START.x;
    const int ly = voxelGridWS.y - INDIRECT_VOLUME_START.y;
    const int lz = voxelGridWS.z - INDIRECT_VOLUME_START.z;
    const int f  = voxelGridWS.w;
    if (f < 0 || f >= PATH_GUIDING_FACE_COUNT)
    {
        return nullptr;
    }
    if (lx < 0 || ly < 0 || lz < 0 || lx >= INDIRECT_VOLUME_SIZE.x || ly >= INDIRECT_VOLUME_SIZE.y || lz >= INDIRECT_VOLUME_SIZE.z)
    {
        return nullptr;
    }
    const size_t lin = static_cast<size_t>(lx)
                     + static_cast<size_t>(INDIRECT_VOLUME_SIZE.x) * ( static_cast<size_t>(ly)
                     + static_cast<size_t>(INDIRECT_VOLUME_SIZE.y) * static_cast<size_t>(lz) );
    const size_t realIndex = INDIRECT_INDEX_VOLUME[lin * PATH_GUIDING_FACE_COUNT + static_cast<size_t>(f)];
    if(realIndex != SIZE_T_MAX)
    {
        return &SCENE_PROBES[realIndex];
    }
    return nullptr;
}

template<bool bUseBVH, bool bUseNEE, bool bUsePathGuiding>
FUNCTION_MODIFIER_DEVICE_INLINE float4 CalculateRadiance(
    curandState* seed, float3 rayOrigin, float3 rayDirection,
    pg::PathGuidingSample* radianceSampleBuffer = nullptr,
    denoise::ScreenGBuffer* screenGBuffer = nullptr, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer = nullptr) // radianceSampleBuffer must be a PATH_GUIDING_COLLECT_DEPTH sized array
{
#ifdef USE_SPECTRUM_RENDERING
    const spectrum::sample::SpectrumSample spectrumLambdaSample = spectrum::sample::SampleCIE1931(Rand3(seed));
    const float spectrumLambda = spectrumLambdaSample.lambda_;
    const float spectrumLambdaPdf = spectrumLambdaSample.pdf_;
    const SpectrumBasis lambdaBasis = spectrum::lut::QueryLambda(spectrumLambda, SPECTRUM_LUT_LAMBDA);
    const float lambdaD65Norm = spectrum::D65Norm(spectrumLambda);
    //spectrum::query::LambdaEncode(&spectrumLambda, lambdaBasis.spectrumBasis_);
    const float3 lambdaXYZ = spectrum::LambdaToCIE1931_XYZ(spectrumLambda);
#else
#endif

#ifdef USE_PATH_GUIDING
    [[maybe_unused]] float3 radianceSampleDirections[PATH_GUIDING_COLLECT_DEPTH];
#ifdef USE_SPECTRUM_RENDERING
    [[maybe_unused]] float radianceSamples[PATH_GUIDING_COLLECT_DEPTH];
    [[maybe_unused]] float radianceMasks[PATH_GUIDING_COLLECT_DEPTH];
#else
    [[maybe_unused]] float3 radianceSamples[PATH_GUIDING_COLLECT_DEPTH];
    [[maybe_unused]] float3 radianceMasks[PATH_GUIDING_COLLECT_DEPTH];
#endif
    [[maybe_unused]] bool radianceValid[PATH_GUIDING_COLLECT_DEPTH];
    [[maybe_unused]] float3 radianceVertexLocation[PATH_GUIDING_COLLECT_DEPTH];
    [[maybe_unused]] float3 radianceVertexDirection[PATH_GUIDING_COLLECT_DEPTH]; // could be normal or reflect vector
    if constexpr (bUsePathGuiding) // TODO: collect non zero NEE sample
    {
        if(radianceSampleBuffer)
        {
#pragma unroll
            for(int i=0;i<PATH_GUIDING_COLLECT_DEPTH;i++)
            {
                radianceSampleBuffer[i] = {}; // clear all radiance sample first, invalid
            }
            //
#pragma unroll
            for(int i=0;i<PATH_GUIDING_COLLECT_DEPTH;i++)
            {
                radianceSampleDirections[i] = {0.0f, 0.0f, 0.0f};
#ifdef USE_SPECTRUM_RENDERING
                radianceSamples[i] = 0.0f;
                radianceMasks[i] = 0.0f;
#else
                radianceSamples[i] = {0.0f, 0.0f, 0.0f};
                radianceMasks[i] = {0.0f, 0.0f, 0.0f};
#endif
                radianceValid[i] = false;
                radianceVertexLocation[i] = {0.0f, 0.0f, 0.0f};
                radianceVertexDirection[i] = {0.0f, 0.0f, 0.0f};
            }
        }
    }
#endif
    //
    int nonDeltaDepth = 0;
    rayDirection = normalize(rayDirection);
    
#ifdef USE_SPECTRUM_RENDERING
    float radiance = 0.0f;
    float mask = 1.0f;
    float3 maskForProbeQuery = {1.0f, 1.0f, 1.0f};
#else
    float3 radiance = {0.0f, 0.0f, 0.0f};
#ifdef USE_3_CHANNEL_PROBE
    float3 maskForProbeQuery = {1.0f, 1.0f, 1.0f};
#endif
    float3 mask = {1.0f, 1.0f, 1.0f};
#endif
    float3 currentRayOrigin = rayOrigin;
    float3 currentRayDirection = rayDirection;
    
    MediaContext currentMediaState = {};
    const OverlapContext firstOverlapContext = intersection::Overlap(currentRayOrigin, SCENE_OBJECTS, SCENE_OBJECT_COUNTS);
    currentMediaState.bInsideMedia_ = firstOverlapContext.bOverlap_;
    currentMediaState.vertexLocation_ = rayOrigin;

    [[maybe_unused]] NEEState previousNEEState = {};
    for(int depth = 0; depth < MAX_TRACING_DEPTH; depth++)
    {
        TraceContext traceContext = Trace<bUseBVH, true /*bool offsetHitPosition*/>(
        nonDeltaDepth,
#ifdef USE_SPECTRUM_RENDERING
        spectrumLambda, lambdaBasis,
#else
#endif
            seed, currentRayOrigin, currentRayDirection, currentMediaState); // already offset
        currentMediaState = traceContext.sampleContext_.mediaContext_; // update media state
        const float usePathGuidingProb = GetPathGuidingProb(traceContext.intersectionContext_.gbuffer_);//lerp(0.0f, PATH_GUIDING_MAX_ALPHA, powf(fminf(1.0f, static_cast<float>(FRAME_INDEX) / PATH_GUIDING_LERP_MAX_FRAME), PATH_GUIDING_LERP_POW));

#ifdef USE_PATH_GUIDING
        [[maybe_unused]] const Octahedron* currentProbePtr = nullptr;
        if constexpr (bUsePathGuiding)
        {
            if ( true && // radianceSampleBuffer no sample buffer so maybe path guiding is not valid skip, if debug use True 
                traceContext.intersectionContext_.bHit_ &&
                (!traceContext.sampleContext_.bDeltaSurface_) &&
                (!traceContext.sampleContext_.bHitPureLight_) &&
                (!currentMediaState.bInsideMedia_) &&
                SCENE_PROBES_COUNTS > 0)
            {
                //currentProbePtr = &SCENE_PROBES[0]; // TODO: query / mix by angle
#ifdef PATH_GUIDING_REFLECT_VECTOR
                // currentRayDirection points toward surface, reflect(currentRayDirection, traceContext.intersectionContext_.hitNormal_) points outward surface
                currentProbePtr = QueryProbe(traceContext.intersectionContext_.hitPosition_, reflect(currentRayDirection, traceContext.intersectionContext_.hitNormal_)); // TODO: query / mix by angle
#else
                currentProbePtr = QueryProbe(traceContext.intersectionContext_.hitPosition_, traceContext.intersectionContext_.hitNormal_); // TODO: query / mix by angle
#endif
                if(currentProbePtr != nullptr)
                {
                    //currentProbePtr->DebugAssertMipmapAverage();
                    const float pBSDFAtNextDirection = traceContext.sampleContext_.nextDirectionPDF_;
#ifdef USE_3_CHANNEL_PROBE
                    const float pProbeAtNextDirection = currentProbePtr->OctahedronHemispherePDF(traceContext.sampleContext_.nextDirection_, traceContext.intersectionContext_.hitNormal_, maskForProbeQuery);
#else
                    const float pProbeAtNextDirection = currentProbePtr->OctahedronHemispherePDF(traceContext.sampleContext_.nextDirection_, traceContext.intersectionContext_.hitNormal_);
#endif
                    if(pBSDFAtNextDirection > 1.0e-12f && pProbeAtNextDirection > 1.0e-12f)
                    {
                        if (Rand1(seed) < usePathGuidingProb)
                        {
                            //const Sample pathGuidedSample = currentProbe.SampleOctahedron(seed);
#ifdef USE_3_CHANNEL_PROBE
                            const Sample pathGuidedSample = currentProbePtr->SampleOctahedronHemisphere(seed, traceContext.intersectionContext_.hitNormal_, maskForProbeQuery);
#else
                            const Sample pathGuidedSample = currentProbePtr->SampleOctahedronHemisphere(seed, traceContext.intersectionContext_.hitNormal_);
#endif
                            if(dot(pathGuidedSample.direction_, traceContext.intersectionContext_.hitNormal_) > 1.0e-12f)
                            {
                                float pBSDFProb = 0.0f;
                                
#ifdef USE_SPECTRUM_RENDERING
                                pBSDFProb = BSDF_PDF(
                                    lambdaBasis, SPECTRUM_LUT_RGB,
#else
                                pBSDFProb = BSDF_PDF(
#endif
                                    traceContext.shadingModel_,
                                    traceContext.intersectionContext_.gbuffer_,
                                    traceContext.intersectionContext_.hitPosition_,
                                    traceContext.intersectionContext_.hitNormal_,
                                    currentRayDirection,
                                    pathGuidedSample.direction_
                                );
                                traceContext.sampleContext_.nextDirection_ = pathGuidedSample.direction_;
#ifdef USE_SPECTRUM_RENDERING
                                traceContext.sampleContext_.hitBRDF_ = EvaluateBSDFWithCos(
                                    spectrumLambda, lambdaBasis, SPECTRUM_LUT_RGB,
#else
                                traceContext.sampleContext_.hitBRDF_ = EvaluateBSDFWithCos(
#endif
                                    traceContext.shadingModel_,
                                    currentRayDirection,
                                    pathGuidedSample.direction_,
                                    traceContext.intersectionContext_,
                                    currentMediaState
                                );
                                traceContext.sampleContext_.nextDirectionPDF_= (1.0f - usePathGuidingProb) * fmaxf(pBSDFProb,1.0e-12f) + usePathGuidingProb * pathGuidedSample.pdf_;
                            }
                            else
                            {
                                currentProbePtr = nullptr;
                            }
                        }
                        else
                        {
                            traceContext.sampleContext_.nextDirectionPDF_ = (1.0f - usePathGuidingProb) * pBSDFAtNextDirection + usePathGuidingProb * pProbeAtNextDirection;
                        }
                    }
                    else
                    {
                        currentProbePtr = nullptr;
                    }
                }
            }
        }
        // collect gbuffer
        if(depth == 0 && screenGBuffer)
        {
            denoise::RecordScreenGBuffer(traceContext, screenGBuffer);
        }
        //
        [[maybe_unused]] float3 endpointLeForGuiding = float3{0,0,0};
#endif
        if (traceContext.intersectionContext_.bHit_ == false) // miss hit
        {
            if(traceContext.sampleContext_.bMissThinSurfaceInsideMedia_ == false)
            {
#ifdef USE_PATH_GUIDING
                if constexpr (bUsePathGuiding)
                {
                    endpointLeForGuiding = traceContext.sampleContext_.hitEmissive_;
                }
#endif
#ifdef USE_SPECTRUM_RENDERING
                radiance = radiance + mask * spectrum::lut::Interact(lambdaBasis, traceContext.sampleContext_.hitEmissive_, SPECTRUM_LUT_RGB) * lambdaD65Norm;
#else
                radiance = radiance + mask * traceContext.sampleContext_.hitEmissive_;
#endif
            }
        }
        else // hit something
        {
            if(traceContext.intersectionContext_.lightIndex_ >= 0) // HIT LIGHT so light index is valid
            {
                bool bAddLe = true;
                float misWeightForLe = 1.0f;
                if constexpr (bUseNEE)
                {
                    if (depth > 0)
                    {
                        if(previousNEEState.previousDidNEE_)
                        {
                            const float pBSDFOmega  = fmaxf(previousNEEState.lastSegmentBSDFPDF_, 0.0f);
                            const float pLightOmega = NEEPDFOmegaFromHit(
                                SCENE_OBJECTS[traceContext.intersectionContext_.objectIndex_], // light object
                                SCENE_LIGHTS_INFO[traceContext.intersectionContext_.lightIndex_], // light info
                                previousNEEState.lastVertexPosition_, // surface point, last is surface, this is light
                                traceContext.intersectionContext_.hitPosition_, // light point
                                traceContext.intersectionContext_.hitNormal_ // light normal
                            );
                            if (pLightOmega >= 0.0f && pBSDFOmega > 0.0f)
                            {
                                misWeightForLe = PowerHeuristic(pBSDFOmega, pLightOmega);
                            }
                            bAddLe = true;
                        }
                        else
                        {
                            bAddLe = true;
                        }
                    }
                }
#ifdef USE_PATH_GUIDING
                if constexpr (bUsePathGuiding)
                {
                    endpointLeForGuiding = traceContext.sampleContext_.hitEmissive_ * misWeightForLe;
                }
#endif
                if (bAddLe)
                {
#ifdef USE_SPECTRUM_RENDERING
                    radiance = radiance + mask * spectrum::lut::Interact(lambdaBasis, traceContext.sampleContext_.hitEmissive_, SPECTRUM_LUT_RGB) * lambdaD65Norm * misWeightForLe;
#else
                    radiance = radiance + mask * traceContext.sampleContext_.hitEmissive_ * misWeightForLe;
#endif
                }
            }
            else
            {
#ifdef USE_PATH_GUIDING
                if constexpr (bUsePathGuiding)
                {
                    if (!allZero(traceContext.sampleContext_.hitEmissive_))
                    {
                        endpointLeForGuiding = traceContext.sampleContext_.hitEmissive_;
                    }
                }
#endif
            }
        }
        
#ifdef USE_PATH_GUIDING
        if constexpr (bUsePathGuiding)
        {
            if (radianceSampleBuffer)
            {
                CollectRadiance(
                    radianceSampleDirections,
                    radianceSamples,
                    radianceMasks,
                    radianceVertexLocation,
                    radianceVertexDirection,
                    radianceValid,
#ifdef USE_SPECTRUM_RENDERING
                    spectrum::lut::Interact(lambdaBasis, endpointLeForGuiding, SPECTRUM_LUT_RGB) * lambdaD65Norm,
#else
                    endpointLeForGuiding,
#endif
                    traceContext.sampleContext_.nextDirection_,
#ifdef USE_SPECTRUM_RENDERING
                    0.0f,
#else
                    /*segmentContrib*/ float3{0.0f, 0.0f, 0.0f},
#endif
                    /*rrSurviveProb*/ 1.0f,
                    /*nonDeltaDepth*/ nonDeltaDepth,
                    /*allowStartAtThisVertex*/ false,
                    /*advanceMasks*/ false,
                    currentRayDirection,
                    traceContext.intersectionContext_.hitPosition_,
                    traceContext.intersectionContext_.hitNormal_,
                    PATH_GUIDING_COLLECT_DEPTH
                );
            }
        }
#endif
        
        if((traceContext.intersectionContext_.bHit_ == false && traceContext.sampleContext_.bMissThinSurfaceInsideMedia_ == false) || traceContext.sampleContext_.bTerminate_ == true) // break
        {
            break;
        }
        
        if(traceContext.sampleContext_.bMissThinSurfaceInsideMedia_ == false) // NEE
        {
            //
            [[maybe_unused]] const EShadingModel currentShadingModel = traceContext.shadingModel_;
            [[maybe_unused]] const GBuffer currentGBuffer = traceContext.intersectionContext_.gbuffer_;
            //
#ifdef USE_SPECTRUM_RENDERING
            const float perceptualLuminance = mask;
#else
            const float perceptualLuminance = dot(mask, float3{0.2126f, 0.7152f, 0.0722f});
#endif
            [[maybe_unused]] float useNEERate = sqrtf(saturate_(perceptualLuminance)); // adhoc
            
            bool bNEEValid = false;
            if constexpr (bUseNEE)
            {
                if(nonDeltaDepth < NEE_MAX_NON_DELTA_DEPTH &&
                    traceContext.sampleContext_.bDeltaSurface_ == false &&
                    currentMediaState.bInsideMedia_ == false &&
                    Rand1(seed) < useNEERate)
                {
                    float3 NEELightDirection = {};
#ifdef USE_SPECTRUM_RENDERING
                    float NEEFCos = {};
                    float NEELe = {};
#else
                    float3 NEEFCos = {};
                    float3 NEELe = {};
#endif
                    float NEEPOmega = 0.0f;
                    // Do NEE
#ifdef USE_SPECTRUM_RENDERING
                    const float radianceNEE = NEE<bUseBVH, bUsePathGuiding>(
                        depth, spectrumLambda, lambdaBasis,
#else
                    const float3 radianceNEE = NEE<bUseBVH, bUsePathGuiding>(
#endif
                        seed,
#ifdef USE_3_CHANNEL_PROBE
                        maskForProbeQuery,
#endif
                        traceContext.shadingModel_,
                        currentRayDirection,
                        traceContext.intersectionContext_,
#ifdef USE_PATH_GUIDING
                        currentProbePtr,
#else
                        nullptr,
#endif
                        &bNEEValid, &NEELightDirection, &NEEFCos, &NEELe, &NEEPOmega
                    );
#ifdef USE_SPECTRUM_RENDERING
                    radiance = radiance + mask * radianceNEE * lambdaD65Norm;
#else
                    radiance = radiance + mask * radianceNEE;
#endif
                }
            }
            previousNEEState.previousDidNEE_ = bNEEValid;
            previousNEEState.lastSegmentBSDFPDF_ = traceContext.sampleContext_.nextDirectionPDF_;
            previousNEEState.lastVertexPosition_ = traceContext.intersectionContext_.hitPosition_;
        }
        
#ifdef USE_SPECTRUM_RENDERING
        const float contrib =
#else
        const float3 contrib =
#endif
            traceContext.sampleContext_.hitBRDF_ / fmaxf(traceContext.sampleContext_.nextDirectionPDF_, 1e-12f);
        //
        [[maybe_unused]] const bool allowStartAtThisVertex =(traceContext.intersectionContext_.bHit_) && (!traceContext.sampleContext_.bDeltaSurface_) && (!traceContext.sampleContext_.bHitPureLight_) && (!currentMediaState.bInsideMedia_);

        float earlyTerminateProbability = 1.0f;
        bool survived = true;
        if (nonDeltaDepth >= MIN_TRACING_DEPTH)
        {
#ifdef USE_SPECTRUM_RENDERING
            const float perceptualLuminance = mask;
            const float maxLuminance = mask;
#else
            const float perceptualLuminance = dot(mask, float3{0.2126f, 0.7152f, 0.0722f});
            const float maxLuminance = max3(mask);
#endif
            const float basicTerminateRate = clamp(perceptualLuminance * 0.5f + maxLuminance * 0.5f, 0.05f, 0.95f);
            earlyTerminateProbability = lerp(basicTerminateRate, 1.0f, currentMediaState.bInsideMedia_ ? DELTA_RR_TERMINATE_REDUCE : 0.0f); // if inside media, lower RR terminate rate
            if (Rand1(seed) > earlyTerminateProbability)
            {
                survived = false;
            }
        }
        if (!survived)
        {
            break;
        }
#ifdef USE_PATH_GUIDING
        if constexpr (bUsePathGuiding)
        {
            if (radianceSampleBuffer)
            {
                CollectRadiance(
                    radianceSampleDirections,
                    radianceSamples,
                    radianceMasks,
                    radianceVertexLocation,
                    radianceVertexDirection,
                    radianceValid,
#ifdef USE_SPECTRUM_RENDERING
                    0.0f,
#else
                    /*endpointLe*/ float3{0.0f, 0.0f, 0.0f},
#endif
                    traceContext.sampleContext_.nextDirection_,
                    contrib,
                    earlyTerminateProbability,
                    nonDeltaDepth,
                    allowStartAtThisVertex,
                    /*advanceMasks*/true,
                    currentRayDirection,
                    traceContext.intersectionContext_.hitPosition_,
                    traceContext.intersectionContext_.hitNormal_,
                    PATH_GUIDING_COLLECT_DEPTH
                );
            }
        }
#endif
        mask = mask * contrib;
        mask = mask / earlyTerminateProbability;
#ifdef USE_3_CHANNEL_PROBE
#ifdef USE_SPECTRUM_RENDERING
        maskForProbeQuery = max(mask * lambdaXYZ, PROBES_QUERY_EPS);
#else
        maskForProbeQuery = max(mask, PROBES_QUERY_EPS);
#endif
        maskForProbeQuery = maskForProbeQuery / (sum3(maskForProbeQuery) + PROBES_QUERY_EPS);
#endif
        //
        currentRayOrigin = traceContext.intersectionContext_.hitPosition_;
        currentRayDirection = traceContext.sampleContext_.nextDirection_;
        if(traceContext.sampleContext_.bDeltaSurface_ == false)
        {
            nonDeltaDepth++;
        }
    }
#ifdef USE_PATH_GUIDING
    if constexpr (bUsePathGuiding)
    {
        if (radianceSampleBuffer)
        {
            int currentWriteIndex = 0;
            for (int i = 0; i < PATH_GUIDING_COLLECT_DEPTH; ++i)
            {
                if(currentWriteIndex >= PATH_GUIDING_COLLECT_DEPTH)
                {
                    break;
                }
                if (!radianceValid[i])
                {
                    continue;
                }
#ifdef USE_3_CHANNEL_PROBE
#ifdef USE_SPECTRUM_RENDERING
                const float3 radianceSampleXYZ = lambdaXYZ * radianceSamples[i] / spectrumLambdaPdf;
                const float strength = fminf(max3(radianceSampleXYZ), PATH_GUIDING_COLLECT_MAX_BRIGHTNESS);
#else
                const float strength = max3(radianceSamples[i]);
#endif
#else
#ifdef USE_SPECTRUM_RENDERING
                //const float3 radianceSampleXYZ = lambdaXYZ * radianceSamples[i] / spectrumLambdaPdf;
                //const float3 radianceSampleRGB = max(XYZ2SRGBLinearD65(radianceSampleXYZ), 0.0f);
                //const float strength = fminf(luminance(radianceSampleRGB), PATH_GUIDING_COLLECT_MAX_BRIGHTNESS); // remember div spectrum lambda pdf
                const float strength = fminf(radianceSamples[i] * lambdaXYZ.y / spectrumLambdaPdf, PATH_GUIDING_COLLECT_MAX_BRIGHTNESS);
#else
                const float strength = fminf(luminance(radianceSamples[i]), PATH_GUIDING_COLLECT_MAX_BRIGHTNESS);
#endif
#endif
                if(strength < PATH_GUIDING_COLLECT_MIN_BRIGHTNESS)
                {
                    continue;
                }
                pg::PathGuidingSample pathGuideSample{};
                pathGuideSample.radianceDirection_ = radianceSampleDirections[i];
#ifdef USE_3_CHANNEL_PROBE
#ifdef USE_SPECTRUM_RENDERING
                pathGuideSample.radiance_ = max(radianceSampleXYZ, 0.0f);
#else
                pathGuideSample.radiance_ = max(radianceSamples[i], 0.0f);
#endif
#else
                pathGuideSample.radianceStrength_ = strength;
#endif
                pathGuideSample.voxelGrid_ = pg::GetVoxelGrid(radianceVertexLocation[i], radianceVertexDirection[i]);
                pathGuideSample.valid_ = true;
                radianceSampleBuffer[currentWriteIndex] = pathGuideSample;
                currentWriteIndex++;
            }
        }
    }
#endif

#ifdef USE_SPECTRUM_RENDERING
    const float3 preOutput = clamp(lambdaXYZ * checkNanInf(radiance) / spectrumLambdaPdf, 0.0f, RADIANCE_CLAMP);
#else
    const float3 preOutput = clamp(checkNanInf(radiance), 0.0f, RADIANCE_CLAMP);
#endif
    
#ifdef DEBUG_PATH_GUIDING_INDIRECT_VOLUME
#ifdef DEBUG_PATH_GUIDING_INDIRECT_VOLUME_DETAILED
    const float4 indirectVolumeColor = TracePathGuidingIndirectVolumeDetailed(rayOrigin, rayDirection, INDIRECT_INDEX_VOLUME, INDIRECT_VOLUME_START, INDIRECT_VOLUME_END, SCENE_PROBES);
#ifdef USE_SPECTRUM_RENDERING
    const float4 output = make_f4(lerp(preOutput, SRGBLinearD65ToXYZ(xyz(indirectVolumeColor)), indirectVolumeColor.w), 1.0f);
#else
    const float4 output = make_f4(lerp(preOutput, xyz(indirectVolumeColor), indirectVolumeColor.w), 1.0f);
#endif
    return output;
#else
    const float4 indirectVolumeColor = TracePathGuidingIndirectVolume(rayOrigin, rayDirection, INDIRECT_INDEX_VOLUME, INDIRECT_VOLUME_START, INDIRECT_VOLUME_END);
    const float4 output = make_f4(lerp(clamp(checkNanInf(radiance), 0.0f, RADIANCE_CLAMP), xyz(indirectVolumeColor), indirectVolumeColor.w * 0.5), 1.0f);
    return output;
#endif
#else
    //
    if(screenStatisticsBuffer)
    {
        denoise::RecordScreenStatisticBuffer(preOutput, screenStatisticsBuffer);
    }
#ifdef USE_SPECTRUM_RENDERING
    const float4 output = make_f4(preOutput, 1.0f); // linear xyz
#else
    const float4 output = make_f4(preOutput, 1.0f); // linear rgb
#endif
    return output;
#endif
}