// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#include "sdf_base.cuh"
#include "sdf_torus.cuh"
#include "sdf_diamond.cuh"

namespace sdf 
{
    // SDFTorus
    template float3 sdfNormalUnit<SDFTorusFunctor>(float3, const SDFInfo&, float, const SDFTorusFunctor&);
    template bool RaymarchLocal<SDFTorusFunctor>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFTorusFunctor&);
    template IntersectionContext IntersectSDFObject<SDFTorusFunctor>(const int, float3, float3, const SceneObject&, const SDFTorusFunctor&);
    template OverlapContext OverlapSDFObject<SDFTorusFunctor>(float3, const SceneObject&, const SDFTorusFunctor&);

    // SDFDiamond
    template float3 sdfNormalUnit<SDFDiamondFunctor>(float3, const SDFInfo&, float, const SDFDiamondFunctor&);
    template bool RaymarchLocal<SDFDiamondFunctor>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFDiamondFunctor&);
    template IntersectionContext IntersectSDFObject<SDFDiamondFunctor>(const int, float3, float3, const SceneObject&, const SDFDiamondFunctor&);
    template OverlapContext OverlapSDFObject<SDFDiamondFunctor>(float3, const SceneObject&, const SDFDiamondFunctor&);

    // SDF Volume
    template float3 sdfNormalUnit<SDFVolumeFunctor>(float3, const SDFInfo&, float, const SDFVolumeFunctor&);
    template bool RaymarchLocal<SDFVolumeFunctor, true>(const int, float3, float3, float, float, float3, const SDFInfo&, float*, float3*, const SDFVolumeFunctor&);
    template IntersectionContext IntersectSDFObject<SDFVolumeFunctor, true>(const int, float3, float3, const SceneObject&, const SDFVolumeFunctor&);
    template OverlapContext OverlapSDFObject<SDFVolumeFunctor>(float3, const SceneObject&, const SDFVolumeFunctor&);
} // namespace sdf