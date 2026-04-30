// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "config.cuh"

#include <vector_types.h>
#include <math_constants.h> // CUDART_INF_F
#include <cuda_runtime.h> // sinf, cosf, expf, powf, sqrtf, rsqrtf, floorf
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
//#include <bit>

#ifdef __CUDA_ARCH__
#define CUDA_UNROLL #pragma unroll
#else
#define CUDA_UNROLL 
#endif

#define CHECK_ERROR() do { auto error = cudaGetLastError(); if (error != 0) std::cout << "CUDA ERROR:" << cudaGetErrorString(error); } while(0)
#define CHECK_CUDA_SYNC() do {\
cudaError_t e = cudaDeviceSynchronize();\
if (e != cudaSuccess) {\
fprintf(stderr,"CUDA sync error %s:%d: %s\n",\
__FILE__, __LINE__, cudaGetErrorString(e));\
fflush(stderr);\
abort();\
}\
} while(0)

#define FUNCTION_MODIFIER __host__ __device__
#define FUNCTION_MODIFIER_INLINE __host__ __device__ __forceinline__
#define FUNCTION_MODIFIER_DEVICE_INLINE __device__ __forceinline__
#define FUNCTION_MODIFIER_DEVICE __device__
#define FUNCTION_MODIFIER_HOST_INLINE __host__ __forceinline__
#define FUNCTION_MODIFIER_STATIC __host__ __device__ static

[[nodiscard]] FUNCTION_MODIFIER_INLINE float sign(float a)
{ 
	if (a > 0.0f)
	{
		return 1.0f;
	}
	if (a < 0.0f)
	{
		return -1.0f;
	}
	return 0.0f;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float wrap01(float v)
{
	float v2 = v - floorf(v);
	return v2;
}
[[nodiscard]] FUNCTION_MODIFIER_INLINE float MirrorAround(const float v, const float center)
{
	const float r = 2.0f * center - v;
	return wrap01(r);
}

[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float inf()
{
#ifdef __CUDA_ARCH__
	return CUDART_INF_F;
#else
	return std::numeric_limits<float>::infinity();
#endif
}

[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float deg2rad(float d)
{
	constexpr float k = CUDART_PI_F / 180.0f;
	return d * k;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 deg2rad(float3 d)
{
	constexpr float k = CUDART_PI_F / 180.0f;
	return {d.x*k, d.y*k, d.z*k};
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float frac(float f) { return f - floorf(f); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float signNotZero(float f) { return f >= 0.0f ? 1.0f : -1.0f; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float saturate_(float v)
{
#ifdef __CUDA_ARCH__
	return __saturatef(v);
#else
	return isnan(v) ? 0.0f : fminf(fmaxf(v, 0.0f), 1.0f);
#endif
}

[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float clamp(float v, float a, float b)
{
	return isnan(v) ? 0.0f : fminf(fmaxf(v, a), b);
}

[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE int clampi(int v, int a, int b)
{
#ifdef __CUDA_ARCH__
	return min(max(v, a), b);
#else
	return std::min(std::max(v, a), b);
#endif
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator+(float2 a, float b) { return { a.x + b, a.y + b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator+(float2 a, float2 b) { return { a.x + b.x, a.y + b.y }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator-(float2 a, float b) { return { a.x - b, a.y - b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator-(float2 a, float2 b) { return { a.x - b.x, a.y - b.y }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator*(float2 a, float b) { return { a.x * b, a.y * b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator*(float2 a, float2 b) { return { a.x * b.x, a.y * b.y }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator/(float2 a, float b) { return { a.x / b,a.y / b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 operator/(float2 a, float2 b) { return { a.x / b.x,a.y / b.y }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 saturate(float2 v) { return { saturate_(v.x),saturate_(v.y) }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 frac(float2 f) { return { frac(f.x), frac(f.y) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 sign(float2 a) { return float2{ sign(a.x), sign(a.y)}; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float2 signNotZero(float2 a) { return float2{ signNotZero(a.x), signNotZero(a.y)}; }
[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float length(float2 a) { return sqrtf(dot(a, a)); }
[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float length2(float2 a) { return dot(a, a); }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator*(float3 a, float b) { return { a.x * b,a.y * b,a.z * b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator*(float3 a, float3 b) { return { a.x * b.x,a.y * b.y,a.z * b.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 hadamard(float3 a, float3 b) { return { a.x * b.x,a.y * b.y,a.z * b.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator*(float b, float3 a) { return a * b; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator/(float3 a, float b) { return { a.x / b,a.y / b,a.z / b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator/(float3 a, float3 b) { return { a.x / b.x,a.y / b.y,a.z / b.z }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator+(float3 a, float b) { return { a.x + b,a.y + b,a.z + b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator+(float3 a, float3 b) { return { a.x + b.x,a.y + b.y,a.z + b.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator+(float b, float3 a) { return a + b; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator-(float3 a, float b) { return { a.x - b,a.y - b,a.z - b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator-(float3 a, float3 b) { return { a.x - b.x,a.y - b.y,a.z - b.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator-(float3 a) { return { -a.x, -a.y, -a.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator-(float b, float3 a) { return { b-a.x, b-a.y, b-a.z }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 xyz(float4 a) { return { a.x, a.y, a.z }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 make_f4(float3 a, float b) { return float4{ a.x, a.y, a.z, b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 make_f4(float x, float y, float z, float w) { return float4{ x, y, z, w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 make_f3(float4 a) { return float3{ a.x, a.y, a.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 make_f3(float x, float y, float z) { return float3{ x, y, z }; }

[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float invSafe(float x)
{
	return (x == 0.0f) ? (x >= 0 ? FLT_MAX : -FLT_MAX) : 1.0f/x;
}
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 inv(float3 a) { return { 1.0f/a.x, 1.0f/a.y, 1.0f/a.z }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 sqrt(float3 a) { return { sqrtf(a.x), sqrtf(a.y), sqrtf(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 invSafe(float3 a) { return { invSafe(a.x), invSafe(a.y), invSafe(a.z) }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 frac(float3 f) { return { frac(f.x), frac(f.y), frac(f.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 log(float3 f) { return { logf(f.x), logf(f.y), logf(f.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 clamp(float3 f, float a, float b) { return { clamp(f.x, a, b), clamp(f.y, a, b), clamp(f.z, a, b) }; }
[[nodiscard]] constexpr FUNCTION_MODIFIER_INLINE float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 saturate(float3 v) { return { saturate_(v.x),saturate_(v.y),saturate_(v.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 max(float3 a, float3 b) { return { fmaxf(a.x, b.x) , fmaxf(a.y, b.y), fmaxf(a.z, b.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 max(float3 a, float b) { return { fmaxf(a.x, b) , fmaxf(a.y, b), fmaxf(a.z, b) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 min(float3 a, float3 b) { return { fminf(a.x, b.x) , fminf(a.y, b.y), fminf(a.z, b.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 min(float3 a, float b) { return { fminf(a.x, b) , fminf(a.y, b), fminf(a.z, b) }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE int3 operator-(int3 a, int3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float fixNanInf(float x, float nan_val, float inf_val)
{
	if (isnan(x))
	{
		return nan_val;
	}
	if (isinf(x))
	{
		return (x > 0.0f ? inf_val : -inf_val);
	}
	return x;
}
[[nodiscard]] FUNCTION_MODIFIER_INLINE bool finitef_(float x)
{
	return !(isnan(x) || isinf(x));
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float checkNanInf(float a)
{
	if (!finitef_(a))
	{
#if !defined(NDEBUG)
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			printf("Invalid number detected: {%.4f}\n", a);
		}
#endif
		return fixNanInf(a, NAN_REPLACEMENT, INF_REPLACEMENT);
	}
	return a;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 checkNanInf(float3 a)
{
	if (!finitef_(a.x) || !finitef_(a.y) || !finitef_(a.z))
	{
#if !defined(NDEBUG)
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			printf("Invalid number detected: {%.4f, %.4f, %.4f}\n", a.x, a.y, a.z);
		}
#endif
		float3 out;
		out.x = fixNanInf(a.x, NAN_REPLACEMENT, INF_REPLACEMENT);
		out.y = fixNanInf(a.y, NAN_REPLACEMENT, INF_REPLACEMENT);
		out.z = fixNanInf(a.z, NAN_REPLACEMENT, INF_REPLACEMENT);
		return out;
	}
	return a;
}

[[nodiscard]] FUNCTION_MODIFIER_DEVICE_INLINE float smooth01(float a, float b, float x)
{
	const float t = saturate_((x - a) / (b - a));
	return t * t * (3.0f - 2.0f * t);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 normalize(float3 v)
{
    float l2 = dot(v,v);
    if(l2 <= 1e-20f)
    {
        return make_f3(0,0,0);
    }
    float inv = rsqrtf(l2);
    return v*inv;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE double lerpd(double a, double b, double v) { return a * (1.0 - v) + b * v; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float lerp(float a, float b, float v) { return a * (1.0f - v) + b * v; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 lerp(float3 a, float3 b, float v){ return { lerp(a.x,b.x, v) ,lerp(a.y,b.y, v) ,lerp(a.z,b.z, v) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 lerp(float3 a, float b, float v){ return { lerp(a.x,b, v) ,lerp(a.y,b, v) ,lerp(a.z,b, v) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 cross(float3 a, float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 pow(float3 a, float b) { return { powf(a.x, b),powf(a.y, b) ,powf(a.z, b) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 exp(float3 a) { return { expf(a.x),expf(a.y) ,expf(a.z) }; };
[[nodiscard]] FUNCTION_MODIFIER_INLINE float max3(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float min3(float3 a) { return fminf(a.x, fminf(a.y, a.z)); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float sum3(float3 a) { return a.x + a.y + a.z; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 sign(float3 a) { return float3{ sign(a.x), sign(a.y),sign(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 signNotZero(float3 a) { return float3{ signNotZero(a.x), signNotZero(a.y),signNotZero(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 abs(float3 a) { return { fabsf(a.x), fabsf(a.y), fabsf(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 sin(float3 a) { return float3{ sinf(a.x),sinf(a.y),sinf(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 cos(float3 a) { return float3{ cosf(a.x),cosf(a.y),cosf(a.z) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE bool allZero(float3 a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE bool allPlus(float3 a) { return a.x > 0.0f && a.y > 0.0f && a.z > 0.0f; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE int3 floorToInt(float3 a) { return {static_cast<int>(floorf(a.x)), static_cast<int>(floorf(a.y)), static_cast<int>(floorf(a.z))};}
[[nodiscard]] FUNCTION_MODIFIER_INLINE int floorToInt(float a) { return static_cast<int>(floorf(a));}
[[nodiscard]] FUNCTION_MODIFIER_INLINE float length(float3 a) { return sqrtf(dot(a, a)); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float length2(float3 a) { return dot(a, a); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float distance(float3 a, float3 b) { return length(a - b); }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float distance2(float3 a, float3 b) { return dot(a, b); }

[[nodiscard]] FUNCTION_MODIFIER_INLINE bool allZero(float a) { return a == 0.0f; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE bool allPlus(float a) { return a > 0.0f; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 refract(float3 inDir, float3 normal, float eta/* n_in / n_out */)
{
	const float3 n = (dot(normal, inDir) > 0.0f) ? -normal : normal;
	const float  cosI = -dot(n, inDir);                      // >= 0
	const float  k = 1.0f - (eta * eta) * fmaxf(0.0f, 1.0f - cosI * cosI);
	const float  cosT = sqrtf(fmaxf(0.0f, k));
	float3 outDir = eta * inDir + (eta * cosI - cosT) * n;
	return outDir;
}

FUNCTION_MODIFIER_INLINE void BuildTangentBasis(const float3 n, float3& t, float3& b)
{
	const float3 a = (fabsf(n.z) < 0.999f) ? float3{0.0f,0.0f,1.0f} : float3{1.0f,0.0f,0.0f};
	t = normalize(cross(a, n));
	b = cross(n, t);
}

FUNCTION_MODIFIER_INLINE void BuildTangentBasisRandom(const float3 n, float3& t, float3& b, const float e) // e ~ [0, 1]
{
	const float3 a = (fabsf(n.z) < 0.999f) ? float3{0.0f,0.0f,1.0f} : float3{1.0f,0.0f,0.0f};
	const float3 t0 = normalize(cross(a, n));
	const float3 b0 = cross(n, t0);
	float s;
	float c;
	sincosf(e * CUDART_PI_F * 2.0f, &s, &c);
	t = c * t0 + s * b0;
	b = -s * t0 + c * b0;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 ToLocal(const float3 v, const float3 t, const float3 b, const float3 n)
{
	return float3{ dot(v,t), dot(v,b), dot(v,n) };
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 ToWorld(const float3 v, const float3 t, const float3 b, const float3 n)
{
	return t * v.x + b * v.y + n * v.z;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float luminance(const float3 c)
{
	return dot(c, float3{0.2126f, 0.7152f, 0.0722f});
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 reflect(const float3 v, const float3 n)
{
	return v - 2.0f * dot(v, n) * n;
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 lerp(float4 a, float4 b, float v){ return { lerp(a.x,b.x, v) ,lerp(a.y,b.y, v) ,lerp(a.z,b.z, v),lerp(a.w,b.w, v) }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 max(float4 a, float b) { return { fmaxf(a.x, b) , fmaxf(a.y, b), fmaxf(a.z, b), fmaxf(a.w, b) }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 min(float4 a, float b) { return { fminf(a.x, b) , fminf(a.y, b), fminf(a.z, b), fminf(a.w, b) }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator*(float4 a, float b) { return { a.x * b,a.y * b,a.z * b, a.w * b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator*(float4 a, float4 b) { return { a.x * b.x,a.y * b.y,a.z * b.z, a.w * b.w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 hadamard(float4 a, float4 b) { return { a.x * b.x,a.y * b.y,a.z * b.z, a.w * b.w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator*(float b, float4 a) { return a * b; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator/(float4 a, float b) { return { a.x / b, a.y / b, a.z / b, a.w / b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator/(float4 a, float4 b) { return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator+(float4 a, float b) { return { a.x + b, a.y + b, a.z + b, a.w + b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator+(float4 a, float4 b) { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator+(float b, float4 a) { return a + b; }

[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator-(float4 a, float b) { return { a.x - b, a.y - b, a.z - b, a.w - b }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator-(float4 a, float4 b) { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator-(float4 a) { return { -a.x, -a.y, -a.z, -a.w }; }
[[nodiscard]] FUNCTION_MODIFIER_INLINE float4 operator-(float b, float4 a) { return { b-a.x, b-a.y, b-a.z, b-a.w }; }

struct float3x3
{
    float3 x;
	float3 y;
	float3 z;
	FUNCTION_MODIFIER_INLINE float3x3() : x{0,0,0}, y{0,0,0}, z{0,0,0} {}
	FUNCTION_MODIFIER_INLINE float3x3(float3 x_, float3 y_, float3 z_) : x(x_), y(y_), z(z_) {}
	
	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 operator*(float3 v) const
	{
		return { dot(x, v), dot(y, v), dot(z, v) };
	}
	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3 mul(float3 v) const
	{
		return *this * v;
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3x3 mul(const float3x3& M) const
	{
		const float3 c0 = make_f3(M.x.x, M.y.x, M.z.x);
		const float3 c1 = make_f3(M.x.y, M.y.y, M.z.y);
		const float3 c2 = make_f3(M.x.z, M.y.z, M.z.z);
		return float3x3(
			make_f3(dot(x,c0), dot(x,c1), dot(x,c2)),
			make_f3(dot(y,c0), dot(y,c1), dot(y,c2)),
			make_f3(dot(z,c0), dot(z,c1), dot(z,c2))
		);
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3x3 operator*(const float3x3& B) const
	{
		return this->mul(B);
	}

	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3x3 transpose() const
	{
		return {
			make_f3(x.x, y.x, z.x),
			make_f3(x.y, y.y, z.y),
			make_f3(x.z, y.z, z.z)
		};
	}
	[[nodiscard]] FUNCTION_MODIFIER_INLINE float3x3 T() const
	{
		return transpose();
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_STATIC float3x3 scale3(float3 s)
	{
		return {
			make_f3(s.x, 0,   0),
			make_f3(0,   s.y, 0),
			make_f3(0,   0,   s.z)
		};
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_STATIC float3x3 rotX(float a)
	{
		const float c = cosf(a);
		const float s = sinf(a);
		return {
			make_f3(1,0,0),
			make_f3(0,c,-s),
			make_f3(0,s, c)
		};
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_STATIC float3x3 rotY(float a)
	{
		const float c = cosf(a);
		const float s = sinf(a);
		return {
			make_f3( c,0,s),
			make_f3( 0,1,0),
			make_f3(-s,0,c)
		};
	}
	
	[[nodiscard]] FUNCTION_MODIFIER_STATIC float3x3 rotZ(float a)
	{
		const float c = cosf(a);
		const float s = sinf(a);
		return {
			make_f3(c,-s,0),
			make_f3(s, c,0),
			make_f3(0, 0,1)
		};
	}
};

[[nodiscard]] FUNCTION_MODIFIER_INLINE uint32_t Pack16x2(const uint16_t a, const uint16_t b)
{
	return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
}
[[nodiscard]] FUNCTION_MODIFIER_INLINE float uintAsFloat(const uint32_t a)
{
#ifdef __CUDA_ARCH__
	return __uint_as_float(a);
#else
    //return std::bit_cast<float>(a);
	float f;
	static_assert(sizeof(f) == sizeof(a), "float and uint32_t must be same size");
	std::memcpy(&f, &a, sizeof(f));
	return f;
#endif
}
FUNCTION_MODIFIER_INLINE void Unpack16x2(const uint32_t v, uint16_t& a, uint16_t& b)
{
	a = static_cast<uint16_t>(v & 0xFFFFu);
	b = static_cast<uint16_t>(v >> 16);
}

[[nodiscard]] FUNCTION_MODIFIER_INLINE uint32_t floatAsUint(const float f)
{
#ifdef __CUDA_ARCH__
	return __float_as_uint(f);
#else
	//return std::bit_cast<uint32_t>(f);
	uint32_t u;
	static_assert(sizeof(u) == sizeof(f), "uint32_t and float must be same size");
	std::memcpy(&u, &f, sizeof(u));
	return u;
#endif
}