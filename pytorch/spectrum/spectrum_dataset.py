from typing import Optional, Tuple, Union, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

import colour
from colour import SpectralShape
from spectrum_sample import sample_cie1931
import random

import itertools
_MEASURED_SDS = []
for chart_name, patches in colour.SDS_COLOURCHECKERS.items():
    for patch_name, sd in patches.items():
        _MEASURED_SDS.append((f"{chart_name}/{patch_name}", sd.copy()))


def _resample_sd_to_shape(sd: colour.SpectralDistribution,
                          shape: colour.SpectralShape,
                          extrapolate: str = "edge") -> Tuple[np.ndarray, np.ndarray]:
    dom_ref = np.arange(shape.start, shape.end + 1e-9, shape.interval, dtype=np.float64)

    sd_c = sd.copy()
    dom_src = np.asarray(sd_c.domain, dtype=np.float64)
    val_src = np.asarray(sd_c.values, dtype=np.float64)

    uniq_idx = np.argsort(dom_src)
    dom_src = dom_src[uniq_idx]
    val_src = val_src[uniq_idx]

    mask = np.isfinite(dom_src) & np.isfinite(val_src)
    dom_src = dom_src[mask]
    val_src = val_src[mask]
    if dom_src.size < 2:
        fill_val = 0.0 if extrapolate == "zero" else (float(val_src[0]) if val_src.size > 0 else 0.0)
        return dom_ref, np.full_like(dom_ref, fill_val, dtype=np.float64)

    if extrapolate == "zero":
        left, right = 0.0, 0.0
    else:
        left, right = float(val_src[0]), float(val_src[-1])

    vals = np.interp(dom_ref, dom_src, val_src, left=left, right=right)
    vals = np.clip(vals, 0.0, 1.0)
    return dom_ref, vals

def _sd_to_rgb_linear(sd, shape):
    dom, vals = _resample_sd_to_shape(sd, shape, extrapolate="edge")
    sd_i = colour.SpectralDistribution(vals, dom)
    D65_sd = colour.SDS_ILLUMINANTS["D65"].copy().interpolate(shape)
    cmfs   = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().interpolate(shape)
    D65 = D65_sd.values.astype(np.float64)
    L   = sd_i.values.astype(np.float64) * D65
    k   = 100.0 / np.sum(D65 * cmfs.values[..., 1])
    X = k * np.sum(L * cmfs.values[..., 0]); 
    Y = k * np.sum(L * cmfs.values[..., 1]); 
    Z = k * np.sum(L * cmfs.values[..., 2])
    XYZ = np.array([X, Y, Z], dtype=np.float64)
    cs = colour.models.RGB_COLOURSPACE_sRGB
    rgb = colour.XYZ_to_RGB(
        XYZ / 100.0,
        cs,
        illuminant=cs.whitepoint,
        chromatic_adaptation_transform=None,
        apply_cctf_encoding=False,
    )
    """
    rgb = colour.XYZ_to_RGB(
    XYZ,
    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
    colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB,
    )
    """
    return np.clip(rgb, 0.0, 1.0), sd_i

def _sample_measured_sd(shape, rng):
    p = rng.random()
    if p < 0.75:
        k = 2
    elif p < 0.9:
        k = 3
    else:
        k = 1
    picks = rng.choice(len(_MEASURED_SDS), size=k, replace=False)
    ws = rng.random(k); ws = ws / np.sum(ws)
    dom = np.arange(shape.start, shape.end + 1e-9, shape.interval, dtype=np.float64)
    v_mix = np.zeros_like(dom, dtype=np.float64)
    for idx, w in zip(picks, ws):
        _, sd = _MEASURED_SDS[idx]
        _, v = _resample_sd_to_shape(sd, shape, extrapolate="edge")
        w = rng.random()
        v_mix += (w * v)
    v_mix = np.clip(v_mix, 0.0, 1.0)
    if rng.random() < 0.75:
        brightness = rng.random() ** 0.75
    else:
        brightness = 1.0
    sd_mix = colour.SpectralDistribution(v_mix * brightness, dom)
    return sd_mix

def _fmt_float_f(v) -> str:
    s = f"{float(v):.9g}"
    if ("." not in s) and ("e" not in s) and ("E" not in s):
        s += ".0"
    return s + "f"

def _emit_array(name: str, values: np.ndarray, per_line: int = 10, indent: str = "    ") -> str:
    lines = []
    row = []
    for i, v in enumerate(values):
        row.append(_fmt_float_f(float(v)))
        if (i + 1) % per_line == 0:
            lines.append(indent + ", ".join(row) + ",")
            row = []
    if row:
        lines.append(indent + ", ".join(row) + ",")
    body = "\n".join(lines)
    return (
        "#ifdef __CUDA_ARCH__\n"
        f"static __device__ constexpr float {name}[D65_COUNT] =\n"
        "#else\n"
        f"inline constexpr float {name}[D65_COUNT] =\n"
        "#endif\n"
        "{\n" + body + "\n};\n"
    )

def make_d65_norm_table(lmin=360.0, lmax=780.0, step=1.0, y_target=1.0):
    shape = colour.SpectralShape(lmin, lmax, step)
    D65_sd = colour.SDS_ILLUMINANTS["D65"].copy().interpolate(shape)  # E_D65(lambda)
    cmfs   = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().interpolate(shape)
    ybar   = cmfs.values[..., 1]                                      # y-bar(lambda)
    E      = D65_sd.values.astype(np.float64)
    dl     = float(shape.interval)
    S      = float(np.sum(E * ybar) * dl)                             # S = sum E*y*dl
    k      = float(y_target / S)
    lambdas = np.arange(lmin, lmax + 1e-9, step, dtype=np.float64)    # for reference
    return lambdas, E.astype(np.float64), S, k

def generate_cuda_d65_norm(lmin=360.0, lmax=780.0, step=1.0, y_target=1.0) -> str:
    # Enforce step == 1.0 to match the reference style that indexes by floor(lambda - MIN).
    if abs(step - 1.0) > 1e-9:
        raise ValueError("This generator matches the reference implementation which assumes step == 1.0.")
    lambdas, E_raw, S, k = make_d65_norm_table(lmin, lmax, step, y_target=y_target)
    n = E_raw.shape[0]
    # Header with constants
    D65_MIN_NM = int(round(lmin))
    D65_MAX_NM = int(round(lmax))
    D65_STEP   = step
    D65_COUNT  = n  # we emit numeric literal for safety

    header = []
    header.append(f"static constexpr int D65_MIN_NM = {D65_MIN_NM};")
    header.append(f"static constexpr int D65_MAX_NM = {D65_MAX_NM};")
    header.append(f"static constexpr float D65_STEP   = {_fmt_float_f(D65_STEP)};")
    header.append(f"static constexpr int D65_COUNT  = {D65_COUNT};")

    # SPD array (raw, not normalized)
    spd_block = _emit_array("D65_SPD", E_raw.astype(np.float64))

    # D65Norm function with linear interpolation and in-function normalization factor k
    k_num = _fmt_float_f(y_target)
    k_den = _fmt_float_f(S)  # show more precision for S
    func = f"""
FUNCTION_MODIFIER_INLINE float D65Norm(float lambda)
{{
    // k normalizes sum(E_D65(lambda)*ybar(lambda)*dl) to y_target.
    constexpr float k = {k_num} / {k_den};
    if (lambda <= D65_MIN_NM)
    {{
        return D65_SPD[0] * k;
    }}
    if (lambda >= D65_MAX_NM)
    {{
        return D65_SPD[D65_COUNT - 1] * k;
    }}
    const float x = lambda - static_cast<float>(D65_MIN_NM);
    const int i = static_cast<int>(std::floor(x));
    const float t = x - static_cast<float>(i);
    const float v0 = D65_SPD[i];
    const float v1 = D65_SPD[i + 1];
    return (v0 + (v1 - v0) * t) * k;
}}
""".strip("\n")

    info = (
        f"// info: l=[{D65_MIN_NM},{D65_MAX_NM}], step={step}, "
        f"S={S:.10f}, k={k:.12f} (y_target={y_target})"
    )
    return "\n".join(header) + "\n" + spd_block + "\n" + func + "\n" + info + "\n"


def _emit_cmf_float3(name: str, xyz: np.ndarray, indent: str = "    ") -> str:
    # xyz shape: [N, 3], ordered as [x_bar, y_bar, z_bar]
    lines = []
    for i in range(xyz.shape[0]):
        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        lines.append(
            f"{indent}float3{{{_fmt_float_f(x)}, {_fmt_float_f(y)}, {_fmt_float_f(z)}}},"
        )
    body = "\n".join(lines)
    return (
        "#ifdef __CUDA_ARCH__\n"
        f"static __device__ constexpr float3 {name}[CIEXYZ_COUNT] =\n"
        "#else\n"
        f"inline constexpr float3 {name}[CIEXYZ_COUNT] =\n"
        "#endif\n"
        "{\n" + body + "\n};\n"
    )

def generate_cuda_ciexyz_cmf(lmin=360.0, lmax=780.0, step=1.0) -> str:
    # Enforce step == 1 nm to match the reference interpolation by index.
    if abs(step - 1.0) > 1e-9:
        raise ValueError("This generator assumes CIEXYZ_STEP_NM == 1.")
    shape = colour.SpectralShape(lmin, lmax, step)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().interpolate(shape)
    # cmfs.values has channels in order [x_bar, y_bar, z_bar]
    xyz = cmfs.values.astype(np.float64)  # shape [N, 3]
    n = xyz.shape[0]

    # Headers (names kept identical to the reference code).
    CIEXYZ_START_NM = int(round(lmin))
    CIEXYZ_END_NM   = int(round(lmax))
    CIEXYZ_STEP_NM  = int(round(step))
    # CIEXYZ_COUNT defined as END - START + 1 to match the reference style.
    header = []
    header.append(f"static constexpr int CIEXYZ_START_NM = {CIEXYZ_START_NM};")
    header.append(f"static constexpr int CIEXYZ_END_NM = {CIEXYZ_END_NM};")
    header.append(f"static constexpr int CIEXYZ_STEP_NM = {CIEXYZ_STEP_NM};")
    header.append("static constexpr int CIEXYZ_COUNT = CIEXYZ_END_NM - CIEXYZ_START_NM + 1;")

    # Table
    table = _emit_cmf_float3("CIEXYZ_CMF", xyz)

    # Interpolation function (kept identical, only range and table differ).
    func = r"""
FUNCTION_MODIFIER_INLINE float3 LambdaToCIE1931_XYZ(const float lambda)
{
    if (lambda <= static_cast<float>(CIEXYZ_START_NM))
    {
        return CIEXYZ_CMF[0];
    }
    if (lambda >= static_cast<float>(CIEXYZ_END_NM))
    {
        return CIEXYZ_CMF[CIEXYZ_COUNT - 1];
    }
    const float u = (lambda - static_cast<float>(CIEXYZ_START_NM)) * (1.0f / static_cast<float>(CIEXYZ_STEP_NM));
    const int i = static_cast<int>(u);
    const float t = u - static_cast<float>(i);
    const float3 a = CIEXYZ_CMF[i];
    const float3 b = CIEXYZ_CMF[i + 1];
    return float3
    {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}
""".strip("\n")

    info = f"// info: l=[{CIEXYZ_START_NM},{CIEXYZ_END_NM}], step={CIEXYZ_STEP_NM}, count={n}, source=CIE 1931 2 Degree Standard Observer"
    return "\n".join(header) + "\n" + table + "\n" + func + "\n" + info + "\n"

def generate_cuda_xyz2srgb_linear_d65_from_colour() -> str:
    M = np.asarray(colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB, dtype=np.float64)

    rows = []
    for r in M:
        rows.append(
            "{{{},{},{}}}".format(_fmt_float_f(r[0]), _fmt_float_f(r[1]), _fmt_float_f(r[2]))
        )
    mat_body = ",\n            ".join(rows)
    ret_line = "return max(float3{0.0f, 0.0f, 0.0f}, rgb);"
    func = f"""
FUNCTION_MODIFIER_INLINE float3 XYZ2SRGBLinearD65(const float3 xyzColor)
{{
    const float3x3 XYZ_2_REC709 = float3x3
    {{
            {mat_body}
    }};
    const float3 rgb = XYZ_2_REC709 * xyzColor;
    {ret_line}
}}
""".strip("\n")
    return func

class SpectralQueryDataset(Dataset):
    def __init__(
        self,
        length: int,
        lambda_bounds: Tuple[float, float] = (360.0, 780.0), # sample lambda min:      376; max: 719.49108887
        lambda_step: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ):
        assert lambda_bounds[0] < lambda_bounds[1], "lambda_bounds must be (min, max)."
        self.length = int(length)
        self.lmin, self.lmax = float(lambda_bounds[0]), float(lambda_bounds[1])
        self.shape = SpectralShape(self.lmin, self.lmax, lambda_step)
        self.rng = rng if rng is not None else np.random.default_rng()
        _ = colour.recovery.RGB_to_sd_Mallett2019([0.5, 0.5, 0.5]).copy().interpolate(self.shape)
        print(f"lambda_bounds: {lambda_bounds}")
        print(f"lambda_step: {lambda_step}")
        print("")
        print("")
        print("// Dd65 Norm")
        print(generate_cuda_d65_norm())
        print("//----------------------------------------------")
        print("")
        print("")
        print("")
        print("")
        print("// CIEXYZ_CMF")
        print(generate_cuda_ciexyz_cmf())
        print("//----------------------------------------------")
        print("")
        print("")
        print("")
        print("")
        print("// XYZ2SRGBLinearD65")
        print(generate_cuda_xyz2srgb_linear_d65_from_colour())
        print("//----------------------------------------------")
        print("")
        print("")

    def __len__(self) -> int:
        return self.length

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        use_measured = (self.rng.random() < 0.9)
        if use_measured and len(_MEASURED_SDS) > 0:
            sd = _sample_measured_sd(self.shape, self.rng)
            rgb, sd_i = _sd_to_rgb_linear(sd, self.shape)
        else:
            rgb = self.rng.random(3, dtype=np.float64) * 1.01 - 0.005
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb = rgb ** 0.75
            rgb = np.clip(rgb, 0.0, 1.0)
            sd = colour.recovery.RGB_to_sd_Mallett2019(rgb)
            sd_i = sd.copy().interpolate(self.shape)
        
        lam = 0.0
        sample_times = 0
        while lam < self.lmin or lam > self.lmax:
            if random.random() < 0.75:
                lam = self.rng.uniform(self.lmin, self.lmax)
            else:
                lam = sample_cie1931() + (random.random() - 0.5) * 4.0
            sample_times = sample_times + 1
            if sample_times >= 3:
                print(f"sample tried {sample_times} times")
        lam = lam + (random.random() - 0.5) * 0.01
        lam = min(max(lam, self.lmin), self.lmax)
        
        
        domain = np.asarray(sd_i.domain, dtype=np.float64)
        values = np.asarray(sd_i.values, dtype=np.float64)
        val = float(np.interp(lam, domain, values))
        if val < 0.0 or val > 1.0:
            print(f"ground truth < 0.0 or > 1.0")
        val = float(np.clip(val, 0.0, 1.0))

        rgb = torch.tensor(rgb, dtype=torch.float32)
        lam = torch.tensor([lam], dtype=torch.float32)
        val = torch.tensor([val], dtype=torch.float32)
        return torch.cat([rgb, lam, val], dim = 0)
    
def roundtrip_check(rgb_linear, sd, step=1.0):
    target = colour.SpectralShape(360, 780, step)
    sd_i = sd.copy().interpolate(target)

    shape_eff = sd_i.shape
    D65_sd = colour.SDS_ILLUMINANTS["D65"].copy().interpolate(shape_eff)
    cmfs   = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().interpolate(shape_eff)

    D65 = D65_sd.values.astype(np.float64)
    L = sd_i.values.astype(np.float64) * D65
    k = 100 / np.sum(D65 * cmfs.values[..., 1])

    X = k * np.sum(L * cmfs.values[..., 0])
    Y = k * np.sum(L * cmfs.values[..., 1])
    Z = k * np.sum(L * cmfs.values[..., 2])
    XYZ = np.array([X, Y, Z], dtype=np.float64)
    
    cs = colour.models.RGB_COLOURSPACE_sRGB
    rgb_rec = colour.XYZ_to_RGB(
        XYZ / 100.0,
        cs,
        illuminant=cs.whitepoint,
        chromatic_adaptation_transform=None,
        apply_cctf_encoding=False,
    )
    rgb_rec = np.clip(rgb_rec, 0.0, 1.0)

    l2   = np.linalg.norm(rgb_rec - rgb_linear)
    maxc = np.max(np.abs(rgb_rec - rgb_linear))
    return rgb_rec, l2, maxc

if __name__ == "__main__":
    ds = SpectralQueryDataset(
        length=100,
        lambda_bounds=(360, 780),
        lambda_step=1.0,
        rng=np.random.default_rng(42),
    )
    for i in range(10000):
        s = ds[i]
        
    for i in range(3):
        s = ds[i]
        print(f"rgb={s[0:3]}, lambda={s[3:4]}nm, value={s[4:5]}")

    K = 10
    l2_list, maxc_list = [], []
    for i in range(K):
        rgb = ds[i][0:3].cpu().numpy().astype(np.float64)
        sd = colour.recovery.RGB_to_sd_Mallett2019(rgb)
        rgb_rec, l2, maxc = roundtrip_check(rgb, sd, step=1.0)
        l2_list.append(l2)
        maxc_list.append(maxc)
        print(f"[{i}] rgb_in={np.round(rgb,4)} -> rgb_rec={np.round(rgb_rec,4)} | L2={l2:.3e}, max|Delta|={maxc:.3e}")

    l2_arr, maxc_arr = np.asarray(l2_list), np.asarray(maxc_list)
    print(f"L2:   mean={l2_arr.mean():.3e}, median={np.median(l2_arr):.3e}, max={l2_arr.max():.3e}")
    print(f"maxDelta: mean={maxc_arr.mean():.3e}, median={np.median(maxc_arr):.3e}, max={maxc_arr.max():.3e}")