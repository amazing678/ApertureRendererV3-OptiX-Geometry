import math
import random
from typing import Sequence

def _fma(a: float, b: float, c: float) -> float:
    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

def _horner(coeffs, x):
    v = coeffs[0]
    for c in coeffs[1:]:
        v = _fma(v, x, c)
    return v

def clamp01(u: float) -> float:
    lo = 0.0
    hi = 1.0
    return lo if u < lo else (hi if u > hi else u)

def ndtrif(u: float) -> float:
    u = clamp01(u)
    uu = u
    a = (-3.969683028665376e+01,
          2.209460984245205e+02,
         -2.759285104469687e+02,
          1.383577518672690e+02,
         -3.066479806614716e+01,
          2.506628277459239e+00)
    b = (-5.447609879822406e+01,
          1.615858368580409e+02,
         -1.556989798598866e+02,
          6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03,
         -3.223964580411365e-01,
         -2.400758277161838e+00,
         -2.549732539343734e+00,
          4.374664141464968e+00,
          2.938163982698783e+00)
    d = ( 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00)
    pLow  = 0.02425
    pHigh = 1.0 - pLow

    if uu < pLow:
        q = math.sqrt(-2.0 * math.log(uu))
        num = _horner((c[0], c[1], c[2], c[3], c[4], c[5]), q)
        den = _horner((d[0], d[1], d[2], d[3], 1.0), q)
        x = num / den
    elif uu > pHigh:
        q = math.sqrt(-2.0 * math.log1p(-uu))
        num = _horner((c[0], c[1], c[2], c[3], c[4], c[5]), q)
        den = _horner((d[0], d[1], d[2], d[3], 1.0), q)
        x = -num / den
    else:
        q = uu - 0.5
        r = q * q
        num = _horner((a[0], a[1], a[2], a[3], a[4], a[5]), r) * q
        den = _horner((b[0], b[1], b[2], b[3], b[4], 1.0), r)
        x = num / den

    INV_SQRT_2PI = 0.3989422804014327
    INV_SQRT2    = 0.7071067811865476
    expo = math.exp(-0.5 * x * x)
    pdf  = expo * INV_SQRT_2PI
    cdf  = 0.5 * math.erfc(-x * INV_SQRT2)
    x   -= (cdf - uu) / pdf
    #delta = (cdf - uu) / pdf
    #x -= delta * (1.0 + 0.5 * x * delta)

    return x


def SampleCIE1931_X(uComp: float, uInvcdf: float) -> float:
    L, U = 360.0, 780.0
    alpha0 = 0.8330258723493694
    # comp 0
    mu0  = 595.7999999999999545
    sig0 = 33.3299999999999983
    Fa0  = 7.4884543e-13
    Fb0  = 0.9999999836707862
    # comp 1
    mu1  = 446.8000000000000114
    sig1 = 19.4400000000000013
    Fa1  = 0.0000040030528583
    Fb1  = 1.0

    if uComp < alpha0:
        t = Fa0 + uInvcdf * (Fb0 - Fa0)
        z = ndtrif(t)
        h = mu0 + sig0 * z
    else:
        t = Fa1 + uInvcdf * (Fb1 - Fa1)
        z = ndtrif(t)
        h = mu1 + sig1 * z

    if h < L: h = L
    if h > U: h = U
    return h

def SampleCIE1931_Y(u: float) -> float:
    L, U = 360.0, 780.0
    mu_p = 6.3269327170812479
    sig  = 0.0750000000000000
    Fa   = 0.0000000020798312
    Fb   = 0.9999953206338883
    t = Fa + u * (Fb - Fa)
    z = ndtrif(t)
    h = math.exp(mu_p + sig * z)
    if h < L: h = L
    if h > U: h = U
    return h

def SampleCIE1931_Z(u: float) -> float:
    L, U = 360.0, 780.0
    mu_p = 6.1114040395252154
    sig  = 0.0510000000000000
    Fa   = 0.0000049890553249
    Fb   = 1.0
    t = Fa + u * (Fb - Fa)
    z = ndtrif(t)
    h = math.exp(mu_p + sig * z)
    if h < L: h = L
    if h > U: h = U
    return h


def LambdaToCIE1931_X(lam: float) -> float:
    kernel1 = (lam - 595.8) / 33.33
    kernel2 = (lam - 446.8) / 19.44
    return 1.065 * math.exp(-0.5 * kernel1 * kernel1) + 0.366 * math.exp(-0.5 * kernel2 * kernel2)

def LambdaToCIE1931_Y(lam: float) -> float:
    LN_556_3 = 6.3213077
    kernel = (math.log(lam) - LN_556_3) / 0.075
    return 1.014 * math.exp(-0.5 * kernel * kernel)

def LambdaToCIE1931_Z(lam: float) -> float:
    LN_449_8 = 6.1088030
    kernel = (math.log(lam) - LN_449_8) / 0.051
    return 1.839 * math.exp(-0.5 * kernel * kernel)

def PdfLambdaToCIE1931_X(lam: float) -> float:
    return LambdaToCIE1931_X(lam) / 106.81109281

def PdfLambdaToCIE1931_Y(lam: float) -> float:
    return LambdaToCIE1931_Y(lam) / 106.34513640

def PdfLambdaToCIE1931_Z(lam: float) -> float:
    return LambdaToCIE1931_Z(lam) / 105.88243587


def sample_cie1931():
    ux, uy, uz = random.random(), random.random(), random.random()
    XImportance, YImportance, ZImportance = 1.0, 1.0, 1.0
    total = XImportance + YImportance + ZImportance
    XRatio = XImportance / total
    YRatio = YImportance / total
    ZRatio = ZImportance / total

    if ux < XRatio:
        lam = SampleCIE1931_X(uy, uz)
    elif ux < XRatio + YRatio:
        lam = SampleCIE1931_Y(uy)
    else:
        lam = SampleCIE1931_Z(uy)

    return lam