// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "random.cuh"
#include "render/sample.cuh"
#include <cuda_runtime.h>

namespace pg
{
    // equal surface mapping
    // https://github.com/erich666/jgt-code/blob/1c80455c8aafe61955f61372380d983ce7453e6d/Volume_13/Number_3/Clarberg2008/mappingfast.h
    FUNCTION_MODIFIER_INLINE float3 UVtoSphereClarberg(const float2 p01)
    {
        // p in [-1,1]^2
        const float u = 2.0f * p01.x - 1.0f;
        const float v = 2.0f * p01.y - 1.0f;
        // rotated square lengths
        const float a = v + u;
        const float b = v - u;
        float r, phi4, zsign; // phi4 in [0,4)
        if (v >= 0.0f)
        {
            if (u >= 0.0f)
            {
                // quadrant 1
                if (a <= 1.0f)
                {
                    r = a; zsign = 1.0f; phi4 = (r==0.0f? 0.0f : v/r);
                }
                else
                {
                    r = 2.0f - a; zsign = -1.0f; phi4 = (1.0f - u) / r;
                }
            }
            else
            {          // quadrant 2
                if (b <= 1.0f)
                {
                    r = b; zsign = 1.0f; phi4 = 1.0f - u / r;
                }
                else
                {
                    r = 2.0f - b; zsign = -1.0f; phi4 = 1.0f + (1.0f - v) / r;
                }
            }
        }
        else
        {
            if (u < 0.0f)
            {  // quadrant 3
                if (a >= -1.0f)
                {
                    r = -a; zsign = 1.0f; phi4 = 2.0f - v / r;
                }
                else
                {
                    r = 2.0f + a; zsign = -1.0f; phi4 = 2.0f + (1.0f + u) / r;
                }
            }
            else
            {          // quadrant 4
                if (b >= -1.0f)
                {
                    r = -b; zsign = 1.0f; phi4 = 3.0f + u / r;
                }
                else
                {
                    r = 2.0f + b; zsign = -1.0f; phi4 = 3.0f + (1.0f + v) / r;
                }
            }
        }
        if (r == 0.0f)
        {
            phi4 = 0.0f;
        }
        const float r2 = r * r;
        const float phi = phi4 * (0.5f * CUDART_PI_F); // = phi4 * (pi/2)
        const float sint = r * sqrtf(fmaxf(0.0f, 2.0f - r2));
        const float cost = zsign * (1.0f - r2);
        float s, c;
        sincosf(phi, &s, &c);
        return float3{ sint * c, sint * s, cost };
    }
    
    // S^2 -> [0,1]^2
    FUNCTION_MODIFIER_INLINE float2 SphereToUVClarberg(const float3 direction)
    {
        const float phi2_over_pi = atan2f(direction.y, direction.x) * (2.0f / CUDART_PI_F);
        float u, v;
        if (direction.z < 0.0f)
        {
            const float r = sqrtf(fmaxf(0.0f, 1.0f + direction.z));
            if (phi2_over_pi >= 0.0f)
            {
                if (phi2_over_pi <= 1.0f)
                {
                    u = 1.0f - r * phi2_over_pi; v = 2.0f - r - u;
                }
                else
                {
                    u = r * (2.0f - phi2_over_pi) - 1.0f; v = 2.0f - r + u;
                }
            }
            else
            {
                if (phi2_over_pi >= -1.0f)
                {
                    u = r * phi2_over_pi + 1.0f; v = r - 2.0f + u;
                }
                else
                {
                    u = r * (2.0f + phi2_over_pi) - 1.0f; v = r - 2.0f - u;
                }
            }
        }
        else
        {
            const float r = sqrtf(fmaxf(0.0f, 1.0f - direction.z));
            if (phi2_over_pi >= 0.0f)
            {
                if (phi2_over_pi < 1.0f)
                {
                    v = r *  phi2_over_pi; u = r - v;
                }
                else
                {
                    v = r * (2.0f - phi2_over_pi); u = v - r;
                }
            }
            else
            {
                if (phi2_over_pi > -1.0f)
                {
                    v = r *  phi2_over_pi; u = r + v;
                }
                else
                {
                    v = -r * (2.0f + phi2_over_pi); u = -(r + v);
                }
            }
        }
        // [-1,1] -> [0,1]
        return float2{ 0.5f * (u + 1.0f), 0.5f * (v + 1.0f) };
    }
    // https://gamedev.stackexchange.com/questions/169508/octahedral-impostors-octahedral-mapping
    FUNCTION_MODIFIER_INLINE float3 UVtoSphereLegacy(const float2 uv) 
    {
        const float2 uvOffset = (uv - 0.5f) * 2.0f;
        float3 position = float3{uvOffset.x, uvOffset.y, 0.0};
        const float2 absolute = {fabsf(position.x), fabsf(position.y)};
        position.z = 1.0f - absolute.x - absolute.y;
        if(position.z < 0) 
        {
            position.x = signNotZero(position.x) * (1.0f - absolute.y);
            position.y = signNotZero(position.y) * (1.0f - absolute.x);
        }
        return normalize(position);
    }
    
    FUNCTION_MODIFIER_INLINE float2 SphereToUVLegacy(const float3 direction)
    {
        const float3 octant = sign(direction);
        const float sum = dot(direction, octant);        
        float3 octahedron = direction / sum;    
        if(octahedron.z < 0)
        {
            const float3 absolute = abs(octahedron);
            octahedron.x = octant.x * (1.0f - absolute.y);
            octahedron.y = octant.y * (1.0f - absolute.x);
        }
        return {octahedron.x * 0.5f + 0.5f, octahedron.y * 0.5f + 0.5f};
    }

    FUNCTION_MODIFIER_INLINE float3 UVtoSphere(const float2 uv)
    {
#ifdef PATH_GUIDING_CLARBERG
        return UVtoSphereClarberg(uv);
#else
        return UVtoSphereLegacy(uv);
#endif
    }

    FUNCTION_MODIFIER_INLINE float2 SphereToUV(const float3 direction)
    {
#ifdef PATH_GUIDING_CLARBERG
        return SphereToUVClarberg(direction);
#else
        return SphereToUVLegacy(direction);
#endif
    }

    FUNCTION_MODIFIER_INLINE float JacobianUVToOctahedronSphereDirection(const float3 sphereDirection)
    {
#ifdef PATH_GUIDING_CLARBERG
        return (4.0f * CUDART_PI_F);
#else
        const float s = fabsf(sphereDirection.x) + fabsf(sphereDirection.y) + fabsf(sphereDirection.z);
        return 4.0f * (s*s*s);
#endif
    }
        
    template<int L = 5> // 1, 2, 4, 8, 16
    struct alignas(16) Texture
    {
        // level 0: 2^(L-1), level 1: 2^(L-2), ..., level L-1: 1
    private:
        static constexpr int CalculateTotalPixelCount()
        {
            static_assert(L > 0, "L must be positive.");
            int sum = 0;
            for (int i = 0; i < L; ++i)
            {
                const int resolution = 1 << i; // 1, 2, 4, ... , 2^(L-1)
                sum += resolution * resolution;
            }
            return sum;
        }
        
    public:
        constexpr static int maxMipmapLevel_ = L;
        constexpr static int totalPixelCount_ = CalculateTotalPixelCount(); // 1 + 4N; 3 + 12N
        constexpr static int totalPixelCountWithPad_ = totalPixelCount_ + 1; // 2 + 4N; 6 + 12N, important for mod 4 alignment
        constexpr static int resolution_ = 1 << (L - 1); // 16
        constexpr static int resolution2_ = resolution_ * resolution_; // 256
#ifdef USE_3_CHANNEL_PROBE
        alignas(16) float4 data_[totalPixelCountWithPad_]; // 2 + 4N; 6 + 12N for better alignment
#else
        alignas(16) float data_[totalPixelCountWithPad_]; // 2 + 4N; 6 + 12N for better alignment
#endif
        unsigned long long count_ = 0; // important, 2 float
        
        FUNCTION_MODIFIER_DEVICE_INLINE void RandomGenerateSphericalDistribution(curandState* seed)
        {
            for(int x=0;x<resolution_;x++)
            {
                for(int y=0;y<resolution_;y++)
                {
                    (*this)(x, y) = Rand1(seed);
                }
            }
            GenerateMipmaps();
        }

        // l=0, off=0; l=1, off=16*16; l=2, off=16*16+8*8
        FUNCTION_MODIFIER_INLINE static int LevelOffset(const int level)
        {
            //static_assert(level > 0);
            //static_assert(level <= L);
            int offset = 0;
            int resolution = 1 << (L - 1); // 16
            CUDA_UNROLL
            for (int i = 0; i < level; ++i)
            {
                offset += resolution * resolution;
                resolution >>= 1;
            }
            return offset;
        }
        FUNCTION_MODIFIER_STATIC float3 ReflectHemisphereNormal(const float3 w, const float3 normal)
        {
            return normalize(w) - 2.0f * dot(normalize(normal), normalize(w)) * normalize(normal);
        }
        // l = 0, res = 32, l = 5, res = 1, off = 32*32+16*16+8*8+4*4+2*2
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_INLINE const float4& Texel(const int x, const int y, int level) const
#else
        FUNCTION_MODIFIER_INLINE const float& Texel(const int x, const int y, int level) const
#endif
        {
            const int resolution = 1 << (L - 1 - level); // 0->16, 1->8
            const int offset = LevelOffset(level);
            return data_[offset + y * resolution + x];
        }
        // stratified sampling, and convert to spherical coord
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_DEVICE_INLINE Sample SampleOctahedron(curandState* seed, const float3 query) const
#else
        FUNCTION_MODIFIER_DEVICE_INLINE Sample SampleOctahedron(curandState* seed) const
#endif
        {
            int currentX = 0;
            int currentY = 0;
            float pTexel = 1.0f;
            for(int level = L - 1; level > 0; --level)
            {
#ifdef USE_3_CHANNEL_PROBE
                const float weightSum = 4.0f * dot(xyz(Texel(currentX, currentY, level)), query);
#else
                const float weightSum = 4.0f * Texel(currentX, currentY, level);
#endif
                const int nextX = currentX << 1;
                const int nextY = currentY << 1;
                const float rand = Rand1(seed);
                if(weightSum <= 0.0f)
                {
                    const int pick = min(3, static_cast<int>(rand * 4.0f)); // 0,1,2,3 25%
                    currentX = nextX + (pick & 1);
                    currentY = nextY + (pick >> 1);
                    pTexel *= 0.25f;
                }
                else
                {
                    float currentCDF = 0.0f;
                    for(int x=0;x<2;x++)
                    {
                        for(int y=0;y<2;y++)
                        {
                            const int currentChildX = nextX + x;
                            const int currentChildY = nextY + y;
#ifdef USE_3_CHANNEL_PROBE
                            const float currentWeight = dot(xyz(Texel(currentChildX, currentChildY, level - 1)), query) / weightSum;
#else
                            const float currentWeight = Texel(currentChildX, currentChildY, level - 1) / weightSum;
#endif
                            currentCDF += currentWeight;
                            if(rand <= currentCDF || (x==1 && y==1))
                            {
                                currentX = currentChildX;
                                currentY = currentChildY;
                                pTexel *= currentWeight;
                                goto Sampled;
                            }
                        }
                    }
                }
                Sampled:
                {
                }
            }
            const float2 rand2 = float2{1.0f, 1.0f} - Rand2(seed);
            const float2 sampledUV = {currentX + rand2.x, currentY + rand2.y}; // [0~32]
            const float2 realUV = saturate(sampledUV / static_cast<float>(resolution_)); // [0~1]
            const float3 direction = UVtoSphere(realUV);
            //const float jacobian = JacobianUVToOctahedronSphere(octahedron);
            const float jacobian = JacobianUVToOctahedronSphereDirection(direction);
            Sample sample;
            sample.direction_ = direction;
            sample.pdf_ = fmaxf(pTexel * resolution2_ / jacobian, 1.0e-12f);
            return sample;
        }
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_DEVICE_INLINE Sample SampleOctahedronHemisphere(curandState* seed, const float3 normal, const float3 query) const
#else
        FUNCTION_MODIFIER_DEVICE_INLINE Sample SampleOctahedronHemisphere(curandState* seed, const float3 normal) const
#endif
        {
#ifdef USE_3_CHANNEL_PROBE
            Sample fullSphereSample = SampleOctahedron(seed, query);
#else
            Sample fullSphereSample = SampleOctahedron(seed);
#endif
            const float3 mirror = ReflectHemisphereNormal(fullSphereSample.direction_, normal);
#ifdef USE_3_CHANNEL_PROBE
            fullSphereSample.pdf_ += OctahedronPDF(mirror, query);
#else
            fullSphereSample.pdf_ += OctahedronPDF(mirror);
#endif
            if(dot(fullSphereSample.direction_, normal) < 0.0f)
            {
                fullSphereSample.direction_ = mirror;
            }
            return fullSphereSample;
        }
        
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_INLINE float OctahedronHemispherePDF(const float3 direction, const float3 normal, const float3 query) const
#else
        FUNCTION_MODIFIER_INLINE float OctahedronHemispherePDF(const float3 direction, const float3 normal) const
#endif
        {
            if (dot(direction, normal) <= 0.0f)
            {
                return 0.0f;
            }
            const float3 mirror = ReflectHemisphereNormal(direction, normal);
#ifdef USE_3_CHANNEL_PROBE
            return OctahedronPDF(direction, query) + OctahedronPDF(mirror, query);
#else
            return OctahedronPDF(direction) + OctahedronPDF(mirror);
#endif
        }
        
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_INLINE float OctahedronPDF(const float3 direction, const float3 query) const
#else
        FUNCTION_MODIFIER_INLINE float OctahedronPDF(const float3 direction) const
#endif
        {
            const float2 uv = SphereToUV(direction);
            int ix = static_cast<int>(floorf(uv.x * resolution_));
            int iy = static_cast<int>(floorf(uv.y * resolution_));
            ix = max(0, min(resolution_ - 1, ix));
            iy = max(0, min(resolution_ - 1, iy));
            
#ifdef USE_3_CHANNEL_PROBE
            const float wLeaf = dot(xyz(Texel(ix, iy, 0)), query);
            const float wAvg = dot(xyz(Texel(0, 0, L - 1)), query);
#else
            const float wLeaf = Texel(ix, iy, 0);
            const float wAvg = Texel(0, 0, L - 1);
#endif
            
            // 1/g = |p|^3/4 = 1/(4 * (|dx|+|dy|+|dz|)^3)
            //const float s = fabsf(direction.x) + fabsf(direction.y) + fabsf(direction.z);
            //const float invG = 1.0f / (4.0f * s * s * s);
            const float invG = 1.0 / JacobianUVToOctahedronSphereDirection(direction);
            if (!(wAvg > 0.0f))
            {
                return invG;
            }
            return fmaxf((wLeaf / wAvg) * invG, 1.0e-12f);
            //const float jacobian = JacobianUVToOctahedronSphereDirection(direction);
            //return (wLeaf / wAvg) / jacobian;
            //pdf
        }
        
#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_INLINE float4& operator()(int x, int y)
#else
        FUNCTION_MODIFIER_INLINE float& operator()(int x, int y)
#endif
        {
            constexpr int finestResolution = 1 << (L - 1);
            return data_[y * finestResolution + x];
        }

#ifdef USE_3_CHANNEL_PROBE
        FUNCTION_MODIFIER_INLINE const float4& operator()(int x, int y) const
#else
        FUNCTION_MODIFIER_INLINE const float& operator()(int x, int y) const
#endif
        {
            constexpr int finestResolution = 1 << (L - 1);
            return data_[y * finestResolution + x];
        }
        
        FUNCTION_MODIFIER_INLINE void GenerateMipmaps() // generate from top to down
        {
            int sourceResolution = resolution_; // level 0
            int sourceOffset = 0;
            int nextOffset = sourceOffset + sourceResolution * sourceResolution;
            
            CUDA_UNROLL
            for (int y = 0; y < resolution_; ++y)
            {
                CUDA_UNROLL
                for (int x = 0; x < resolution_; ++x)
                {
#ifdef USE_3_CHANNEL_PROBE
                    data_[x + y * resolution_] = max(data_[x + y * resolution_], PATH_GUIDING_PROBE_MIN_BRIGHTNESS); // prevent pdf too small
#else
                    data_[x + y * resolution_] = fmaxf(data_[x + y * resolution_], PATH_GUIDING_PROBE_MIN_BRIGHTNESS); // prevent pdf too small
#endif
                }
            }
            // gen level 1, 2, ..., L-1
            CUDA_UNROLL
            for (int level = 0; level < L - 1; ++level)
            {
                const int destinationResolution = sourceResolution >> 1;
                const int destinationOffset = nextOffset;

                for (int y = 0; y < destinationResolution; ++y)
                {
                    const int sourceY = y << 1; // * 2
                    for (int x = 0; x < destinationResolution; ++x)
                    {
                        const int sourceX = x << 1; // * 2

#ifdef USE_3_CHANNEL_PROBE
                        const float4 sampleA = data_[sourceOffset + (sourceY + 0) * sourceResolution + (sourceX + 0)];
                        const float4 sampleB = data_[sourceOffset + (sourceY + 0) * sourceResolution + (sourceX + 1)];
                        const float4 sampleC = data_[sourceOffset + (sourceY + 1) * sourceResolution + (sourceX + 0)];
                        const float4 sampleD = data_[sourceOffset + (sourceY + 1) * sourceResolution + (sourceX + 1)];
#else
                        const float sampleA = data_[sourceOffset + (sourceY + 0) * sourceResolution + (sourceX + 0)];
                        const float sampleB = data_[sourceOffset + (sourceY + 0) * sourceResolution + (sourceX + 1)];
                        const float sampleC = data_[sourceOffset + (sourceY + 1) * sourceResolution + (sourceX + 0)];
                        const float sampleD = data_[sourceOffset + (sourceY + 1) * sourceResolution + (sourceX + 1)];
#endif

                        data_[destinationOffset + y * destinationResolution + x] = (sampleA + sampleB + sampleC + sampleD) * 0.25f;
                    }
                }
                sourceOffset = destinationOffset;
                sourceResolution = destinationResolution;
                nextOffset = sourceOffset + sourceResolution * sourceResolution;
            }
            CUDA_UNROLL
            for(int i=totalPixelCount_; i < totalPixelCountWithPad_; ++i)
            {
#ifdef USE_3_CHANNEL_PROBE
                data_[i] = float4{0.0f, 0.0f, 0.0f, 0.0f};
#else
                data_[i] = 0.0f;
#endif
            }
        }
        
        FUNCTION_MODIFIER_INLINE void DebugAssertMipmapAverage() const
        {
            double sum = 0.0;
            for (int y=0; y<resolution_; ++y)
            {
                for (int x=0; x<resolution_; ++x)
                {
                    sum += static_cast<double>(Texel(x, y, 0));
                }
            }
            const double avg0 = sum / static_cast<double>(resolution2_);
            const double top  = static_cast<double>(Texel(0, 0, maxMipmapLevel_ - 1));
            if (fabs(avg0 - top) > 1e-6 * fmax(1.0, fabs(avg0)))
            {
                printf("Mipmap not generated, MAE error: %.8lf\n", fabs(avg0 - top));
            }
        }
        
    };
}

using Octahedron = pg::Texture<PATH_GUIDING_RESOLUTION_LEVEL>;
static_assert(alignof(Octahedron) >= 16, "Texture must be 16B aligned");
static_assert(sizeof(Octahedron) % 16 == 0, "Texture size must be multiple of 16");
static_assert(offsetof(Octahedron, data_) % 16 == 0, "data_ must start at 16B boundary");