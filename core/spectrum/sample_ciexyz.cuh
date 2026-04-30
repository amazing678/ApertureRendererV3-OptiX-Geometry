// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

#include "vector.cuh"
#include "ciexyz.cuh"

namespace spectrum
{
    namespace sample
    {
        struct SpectrumSample
        {
            float lambda_ = 0.0f;
            float pdf_ = 0.0f;
        };
    
        FUNCTION_MODIFIER_INLINE float Clamp01OpenF(const float u) 
        {
            const float lo = nextafterf(0.0f, 1.0f);
            const float hi = nextafterf(1.0f, 0.0f);
            return u < lo ? lo : (u > hi ? hi : u);
        }

        // ndtri (float)
        template<bool bNewton = true>
        FUNCTION_MODIFIER_INLINE float ndtrif(float u)
        {
            //u = Clamp01OpenF(u);
            u = clamp(u, 0.0f, 1.0f);
            const float uu = u;
            constexpr float sign = 1.0f;
            /*
            float sign = 1.0f;
            float uu = u;
            if (uu > 0.5f)
            {
                uu = fmaf(-uu, 1.0f, 1.0f);
                sign = -1.0f;
            }
            */
            constexpr float a[] =
            {
                -3.969683028665376e+01f,
                 2.209460984245205e+02f,
                -2.759285104469687e+02f,
                 1.383577518672690e+02f,
                -3.066479806614716e+01f,
                 2.506628277459239e+00f
            };
            constexpr float b[] =
            {
                -5.447609879822406e+01f,
                 1.615858368580409e+02f,
                -1.556989798598866e+02f,
                 6.680131188771972e+01f,
                -1.328068155288572e+01f
            };
            constexpr float c[] =
            {
                -7.784894002430293e-03f,
                -3.223964580411365e-01f,
                -2.400758277161838e+00f,
                -2.549732539343734e+00f,
                 4.374664141464968e+00f,
                 2.938163982698783e+00f
            };
            constexpr float d[] =
            {
                7.784695709041462e-03f,
                3.224671290700398e-01f,
                2.445134137142996e+00f,
                3.754408661907416e+00f
           };
            constexpr float pLow = 0.02425f;
            constexpr float pHigh = 1.0f - pLow;

            float x;
            if (uu < pLow) 
            {
                const float q = sqrtf(-2.0f * logf(uu));
                float num = fmaf(c[0], q, c[1]);
                num = fmaf(num, q, c[2]);
                num = fmaf(num, q, c[3]);
                num = fmaf(num, q, c[4]);
                num = fmaf(num, q, c[5]);
                float den = fmaf(d[0], q, d[1]);
                den = fmaf(den, q, d[2]);
                den = fmaf(den, q, d[3]);
                den = fmaf(den, q, 1.0f);
                x = num / den;
            }
            else if (uu > pHigh) // will not trigger
                {
                const float q = sqrtf(-2.0f * log1pf(-uu));
                float num = fmaf(c[0], q, c[1]);
                num = fmaf(num, q, c[2]);
                num = fmaf(num, q, c[3]);
                num = fmaf(num, q, c[4]);
                num = fmaf(num, q, c[5]);
                float den = fmaf(d[0], q, d[1]);
                den = fmaf(den, q, d[2]);
                den = fmaf(den, q, d[3]);
                den = fmaf(den, q, 1.0f);
                x = -num / den;
                }
            else
            {
                const float q = uu - 0.5f;
                const float r = q * q;
                float num = fmaf(a[0], r, a[1]);
                num = fmaf(num, r, a[2]);
                num = fmaf(num, r, a[3]);
                num = fmaf(num, r, a[4]);
                num = fmaf(num, r, a[5]);
                num *= q;
                float den = fmaf(b[0], r, b[1]);
                den = fmaf(den, r, b[2]);
                den = fmaf(den, r, b[3]);
                den = fmaf(den, r, b[4]);
                den = fmaf(den, r, 1.0f);
                x = num / den;
            }
            if constexpr (bNewton)
            {
                constexpr float INV_SQRT_2PI = 0.3989422804014327f;
                constexpr float INV_SQRT2 = 0.7071067811865476f;
                const float expo = expf(fmaf(-0.5f, x * x, 0.0f));
                const float pdf = expo * INV_SQRT_2PI;
                const float cdf = 0.5f * erfcf(-x * INV_SQRT2);
                x -= (cdf - uu) / pdf;
                // Halley:
                //const float delta = (cdf - uu) / pdf;
                //x -= delta * (1.0f + 0.5f * x * delta);
                /*
                const float pdf = expf(fmaf(-0.5f, x * x, 0.0f)) * INV_SQRT_2PI;
                float diff;
                if (uu > 0.5f)
                {
                    const float Q = 0.5f * erfcf(x * INV_SQRT2);
                    const float oneMinusU = fmaf(-uu, 1.0f, 1.0f);
                    diff = oneMinusU - Q;// = (1 - uu) - Q(x)
                }
                else
                {
                    const float Phi = 0.5f * erfcf(-x * INV_SQRT2);
                    diff = Phi - uu;
                }
                const float delta = diff / pdf;
                // Halley
                x -= delta * (1.0f + 0.5f * x * delta);
                */
            }
            return sign * x;
        }

        FUNCTION_MODIFIER_INLINE float SampleCIE1931_X(const float uComp, const float uInvcdf)
        {
            constexpr float L = 360.0f;
            constexpr float U = 780.0f;
            constexpr float alpha0 = 0.8330258723493694f;
            // comp 0
            constexpr float mu0 = 595.7999999999999545f;
            constexpr float sig0 = 33.3299999999999983f;
            constexpr float Fa0 = 7.4884543e-13f;
            constexpr float Fb0 = 0.9999999836707862f;

            // comp 1
            constexpr float mu1 = 446.8000000000000114f;
            constexpr float sig1 = 19.4400000000000013f;
            constexpr float Fa1 = 0.0000040030528583f;
            constexpr float Fb1 = 1.0f; // nextafterf(1.0f, 0.0f);

            float h;
            if (uComp < alpha0)
            {
                const float t0 = Fa0 + uInvcdf * (Fb0 - Fa0);
                const float z0 = ndtrif(t0);
                h = mu0 + sig0 * z0;
            }
            else
            {
                const float t1 = Fa1 + uInvcdf * (Fb1 - Fa1);
                const float z1 = ndtrif(t1);
                h = mu1 + sig1 * z1;
            }
            if (h < L)
            {
                h = L;
            }
            if (h > U)
            {
                h = U;
            }
            return h;
        }

        FUNCTION_MODIFIER_INLINE float SampleCIE1931_Y(const float u)
        {
            constexpr float L = 360.0f;
            constexpr float U = 780.0f;
            constexpr float mu_p = 6.3269327170812479f;
            constexpr float sig = 0.0750000000000000f;
            constexpr float Fa = 0.0000000020798312f;
            constexpr float Fb = 0.9999953206338883f;
            const float t = Fa + u * (Fb - Fa);
            const float z = ndtrif(t);
            float h = expf(mu_p + sig * z);
            if (h < L)
            {
                h = L;
            }
            if (h > U)
            {
                h = U;
            }
            return h;
        }

        FUNCTION_MODIFIER_INLINE float SampleCIE1931_Z(const float u)
        {
            constexpr float L = 360.0f;
            constexpr float U = 780.0f;
            constexpr float mu_p = 6.1114040395252154f;
            constexpr float sig = 0.0510000000000000f;
            constexpr float Fa = 0.0000049890553249f;
            constexpr float Fb = 1.0f; // nextafterf(1.0f, 0.0f);
            const float t = Fa + u * (Fb - Fa);
            const float z = ndtrif(t);
            float h = expf(mu_p + sig * z);
            if (h < L)
            {
                h = L;
            }
            if (h > U)
            {
                h = U;
            }
            return h;
        }
    
        FUNCTION_MODIFIER_INLINE float LambdaToCIE1931_X_Approximated(const float lambda)
        {
            const float kernel1 = (lambda - 595.8f) / 33.33f;
            const float kernel2 = (lambda - 446.8f) / 19.44f;
            return 1.065f * expf(-0.5f * kernel1 * kernel1) + 0.366f * expf(-0.5f * kernel2 * kernel2);
        }
    
        FUNCTION_MODIFIER_INLINE float LambdaToCIE1931_Y_Approximated(const float lambda)
        {
            constexpr float LN_556_3 = 6.3213077f;
            const float kernel = (logf(lambda) - LN_556_3) / 0.075f;
            return 1.014f * expf(-0.5f * kernel * kernel);
        }
    
        FUNCTION_MODIFIER_INLINE float LambdaToCIE1931_Z_Approximated(const float lambda)
        {
            constexpr float LN_449_8 = 6.1088030f;
            const float kernel = (logf(lambda) - LN_449_8) / 0.051f;
            return 1.839f * expf(-0.5f * kernel * kernel);
        }
    
        FUNCTION_MODIFIER_INLINE float PdfLambdaToCIE1931_X(const float lambda)
        {
            return LambdaToCIE1931_X_Approximated(lambda) / 106.81109281f;
        }
    
        FUNCTION_MODIFIER_INLINE float PdfLambdaToCIE1931_Y(const float lambda)
        {
            return LambdaToCIE1931_Y_Approximated(lambda) / 106.34513640f;
        }
    
        FUNCTION_MODIFIER_INLINE float PdfLambdaToCIE1931_Z(const float lambda)
        {
            return LambdaToCIE1931_Z_Approximated(lambda) / 105.88243587f;
        }
        FUNCTION_MODIFIER_INLINE SpectrumSample SampleCIE1931_Simple(const float3 rand)
        {
            SpectrumSample result{};
            result.lambda_ = rand.x * (780.0f - 360.0f) + 360.0f;
            result.pdf_ = 1.0f / (780.0f - 360.0f);
            return result;
        }
        // sample lambda min: 361; max: 774.40490723
        FUNCTION_MODIFIER_INLINE SpectrumSample SampleCIE1931(const float3 rand)
        {
            //return {rand.x * (780.0f - 360.0f) + 360.0f, 1.0f/(780.0f - 360.0f)};
            constexpr float uniformImportance = 0.1f;
            constexpr float XImportance = 1.0f;
            constexpr float YImportance = 1.0f;
            constexpr float ZImportance = 1.0f;
            constexpr float total = uniformImportance + XImportance + YImportance + ZImportance;
            constexpr float uniformRatio = uniformImportance / total;
            constexpr float XRatio = XImportance / total;
            constexpr float YRatio = YImportance / total;
            constexpr float ZRatio = ZImportance / total;
            //
            SpectrumSample result{};
            float lambda;
            if (rand.x < uniformRatio)
            {
                lambda = rand.y * (780.0f - 360.0f) + 360.0f;
            }
            else if (rand.x < uniformRatio + XRatio)
            {
                lambda = SampleCIE1931_X(rand.y, rand.z);
            }
            else if (rand.x < uniformRatio + XRatio + YRatio)
            {
                lambda = SampleCIE1931_Y(rand.y);
            }
            else
            {
                lambda = SampleCIE1931_Z(rand.y);
            }
            const float pMix =  uniformRatio * 1.0f/(780.0f - 360.0f) +
                                XRatio * PdfLambdaToCIE1931_X(lambda) +
                                YRatio * PdfLambdaToCIE1931_Y(lambda) +
                                ZRatio * PdfLambdaToCIE1931_Z(lambda);
            result.lambda_ = lambda;
            result.pdf_ = fmaxf(pMix, 1e-16f);
            return result;
        }
    
        FUNCTION_MODIFIER_INLINE float3 LambdaToCIE1931_XYZ_Approximated(const float lambda) // TODO: sample use single lobe, eval use multi lobe
        {
            return float3{LambdaToCIE1931_X_Approximated(lambda), LambdaToCIE1931_Y_Approximated(lambda), LambdaToCIE1931_Z_Approximated(lambda)};
        }
    
        FUNCTION_MODIFIER_HOST_INLINE void ValidationSampleCIEXYZ()
        {
            printf("ndtrif(0.5f): %.6f\n", spectrum::sample::ndtrif(0.5f));
            printf("ndtrif(0.02425f): %.6f\n", spectrum::sample::ndtrif(0.02425f));
            printf("ndtrif(0.97575f): %.6f\n", spectrum::sample::ndtrif(0.97575f));
            printf("ndtrif(1.0e-6f): %.6f\n", spectrum::sample::ndtrif(1.0e-6f));
            printf("ndtrif(1.0f - (1.0e-6f)): %.6f\n", spectrum::sample::ndtrif(1.0f - (1.0e-6f)));
    
            auto Phi = [](const float x)
            {
                return 0.5f * erfcf(-x * 0.7071067811865476f); // 0.5*erfc(-x/sqrt2)
            };
            float maxAbsErr = 0.0f;
            for (int i = 0; i < 1000000; ++i)
            {
                float u = (static_cast<float>(i) + 0.5f) / 1000001.0f;
                float x = spectrum::sample::ndtrif<true>(u);
                const float uprime = Phi(x);
                maxAbsErr = fmaxf(maxAbsErr, fabsf(u - uprime));
            }
            printf("ndtrif round-trip max |u-u'| = %.9g\n", maxAbsErr);
            //
            constexpr float L = 360.0f;
            constexpr float U = 780.0f;
            const auto XFit = [](const double lambda)->double
            {
                const double k1 = (lambda - 595.8) / 33.33;
                const double k2 = (lambda - 446.8) / 19.44;
                return 1.065 * std::exp(-0.5 * k1 * k1) + 0.366 * std::exp(-0.5 * k2 * k2);
            };
            const auto YFit = [](const double lambda)->double
            {
                const double k = (std::log(lambda) - std::log(556.3)) / 0.075;
                return 1.014 * std::exp(-0.5 * k * k);
            };
            const auto ZFit = [](const double lambda)->double
            {
                const double k = (std::log(lambda) - std::log(449.8)) / 0.051;
                return 1.839 * std::exp(-0.5 * k * k);
            };
            const auto Integral = [&](auto&& F)->double
            {
                constexpr int M = 10000000;
                const float a = L;
                const float b = U;
                const double h = (b - a) / M;
                double acc = 0.5 * (F(a) + F(b));
                for (int i = 1; i < M; ++i)
                {
                    const double x = a + h * i;
                    acc += F(x);
                }
                return acc * h;
            };
        
            const double XTotal = Integral(XFit);
            const double YTotal = Integral(YFit);
            const double ZTotal = Integral(ZFit);
            printf("XTotal: %.8lf\n", XTotal);
            printf("YTotal: %.8lf\n", YTotal);
            printf("ZTotal: %.8lf\n", ZTotal);

            const auto pX = [&XFit, &XTotal](const double lambda)->double
            {
                return XFit(lambda) / XTotal;
            };
            const auto pY = [&YFit, &YTotal](const double lambda)->double
            {
                return YFit(lambda) / YTotal;
            };
            const auto pZ = [&ZFit, &ZTotal](const double lambda)->double
            {
                return ZFit(lambda) / ZTotal;
            };

            struct NumericCDF1D
            {
                std::vector<double> xs, Fs;
                double L=0.0, U=0.0;
                double operator()(double x) const
                {
                    if (x <= xs.front())
                    {
                        return 0.0;
                    }
                    if (x >= xs.back())
                    {
                        return 1.0;
                    }
                    const auto it = std::lower_bound(xs.begin(), xs.end(), x);
                    const int j = static_cast<int>(it - xs.begin());
                    const double x0 = xs[j-1], x1 = xs[j];
                    const double f0 = Fs[j-1], f1 = Fs[j];
                    const double t = (x - x0) / (x1 - x0);
                    return f0 + t * (f1 - f0);
                }
            };
            auto MakeNumericCDF = [&](auto&& pdf, double a, double b, int M)->NumericCDF1D
            {
                NumericCDF1D c; c.L=a; c.U=b; c.xs.resize(M+1); c.Fs.resize(M+1);
                const double h = (b - a) / M;
                auto f_at = [&](double x){ double v = pdf(x); return v > 0.0 ? v : 0.0; };

                c.xs[0] = a; c.Fs[0] = 0.0;
                double prevf = f_at(a), cum = 0.0;
                for (int i = 1; i <= M; ++i)
                {
                    const double x = a + h * i;
                    c.xs[i] = x;
                    const double fi = f_at(x);
                    cum += 0.5 * (prevf + fi) * h;
                    c.Fs[i] = cum;
                    prevf = fi;
                }
                const double total = cum > 1e-300 ? cum : 1e-300;
                for (int i = 0; i <= M; ++i) c.Fs[i] /= total;
                c.Fs.back() = 1.0;
                return c;
            };
            auto KS = [&](std::vector<double>& u)->std::pair<double,double>
            {
                const int n = static_cast<int>(u.size());
                std::sort(u.begin(), u.end());
                double d_plus = 0.0, d_minus = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    const double ui = u[i];
                    const double fi = (i + 1) / static_cast<double>(n); // F_n(x_i+)
                    const double gi = (i) / static_cast<double>(n); // F_n(x_i-)
                    d_plus = max(d_plus,  fi - ui);
                    d_minus = max(d_minus, ui - gi);
                }
                const double Dn = max(d_plus, d_minus);
                const double sqn = std::sqrt(static_cast<double>(n));
                const double t = (sqn + 0.12 + 0.11 / sqn) * Dn; // Massey
                double p = 0.0;
                for (int k = 1; k <= 100000; ++k)
                {
                    const double term = std::exp(-2.0 * k * k * t * t);
                    p += (k & 1) ? (2.0 * term) : (-2.0 * term);
                    if (term < 1e-12) break;
                }
                p = std::clamp(p, 0.0, 1.0);
                return {Dn, p};
            };
            auto KSCheckCDF = [&](const char* tag, const NumericCDF1D& cdf, auto&& sampler, const int N)
            {
                std::vector<double> u; u.reserve(N);
                for (int i = 0; i < N; ++i)
                {
                    constexpr double eps = 1e-12;
                    const double x  = static_cast<double>(sampler());
                    double ui = cdf(x);
                    ui = min(max(ui, eps), 1.0 - eps);
                    u.push_back(ui);
                }
                auto [Dn, p] = KS(u);
                printf("[KS %s] D_n = %.6g, p = %.6g  %s\n", tag, Dn, p, (p < 0.05 ? "<= REJECT" : "OK"));
            };
            const auto RNG01 = []()->double
            {
                static uint64_t s = 0x9E3779B97F4A7C15ull;
                s = s * 2862933555777941757ull + 3037000493ull;
                return ((s >> 11) * (1.0 / 9007199254740992.0));
            };
            const auto RNG01F = [&RNG01]()->float
            {
                return static_cast<float>(RNG01());
            };
            constexpr int MCdf = 1000000;
            const auto FXNum = MakeNumericCDF(pX, L, U, MCdf);
            const auto FYNum = MakeNumericCDF(pY, L, U, MCdf);
            const auto FZNum = MakeNumericCDF(pZ, L, U, MCdf);
            printf("PDF normalization (X,Y,Z) ~= (%.9f, %.9f, %.9f)\n", FXNum.Fs.back(), FYNum.Fs.back(), FZNum.Fs.back());
            constexpr int Nks = 1000000;
            KSCheckCDF("X", FXNum, [&](){ return static_cast<double>(spectrum::sample::SampleCIE1931_X(RNG01F(), RNG01F())); }, Nks);
            KSCheckCDF("Y", FYNum, [&](){ return static_cast<double>(spectrum::sample::SampleCIE1931_Y(RNG01F())); }, Nks);
            KSCheckCDF("Z", FZNum, [&](){ return static_cast<double>(spectrum::sample::SampleCIE1931_Z(RNG01F())); }, Nks);

            float minLambda = FLT_MAX;
            float maxLambda = -FLT_MAX;
            for(int i=0;i<1000000;i++)
            {
                const SpectrumSample spectrum = SampleCIE1931({RNG01F(), RNG01F(), RNG01F()});
                minLambda = std::min(minLambda, spectrum.lambda_);
                maxLambda = std::max(maxLambda, spectrum.lambda_);
            }
            printf("sample lambda min: %8.f; max: %.8f\n", minLambda, maxLambda);
            // sample lambda min:      376; max: 719.49108887
            const int pdfN = 100000;
            {
                float pdf = 0.0f;
                for(int i=0;i<pdfN;i++)
                {
                    pdf += PdfLambdaToCIE1931_X(RNG01F() * (780.0f - 360.0f) + 360.0f) * (780.0f - 360.0f);
                }
                pdf /= pdfN;
                printf("PdfLambdaToCIE1931_X integral: %.4f\n", pdf);
            }
            {
                float pdf = 0.0f;
                for(int i=0;i<pdfN;i++)
                {
                    pdf += PdfLambdaToCIE1931_Y(RNG01F() * (780.0f - 360.0f) + 360.0f) * (780.0f - 360.0f);
                }
                pdf /= pdfN;
                printf("PdfLambdaToCIE1931_Y integral: %.4f\n", pdf);
            }
            {
                float pdf = 0.0f;
                for(int i=0;i<pdfN;i++)
                {
                    pdf += PdfLambdaToCIE1931_Z(RNG01F() * (780.0f - 360.0f) + 360.0f) * (780.0f - 360.0f);
                }
                pdf /= pdfN;
                printf("PdfLambdaToCIE1931_Z integral: %.4f\n", pdf);
            }
        }
    }
}