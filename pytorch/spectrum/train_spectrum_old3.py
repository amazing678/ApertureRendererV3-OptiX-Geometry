import os
import math
import argparse
import random
import numpy as np
from time import time
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import colour
from colour import SpectralShape

from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from spectrum_dataset import SpectralQueryDataset


import contextlib
import copy

def param_groups_decay(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_optimizer(model, optim_name: str, lr: float, weight_decay: float):
    groups = param_groups_decay(model, weight_decay)
    if optim_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    elif optim_name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.03, min_lr_ratio: float = 0.01):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p, alpha=1.0 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_new_folder_path(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    subfolders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()
    ]
    max_number = -1
    for folder in subfolders:
        if not folder.isdigit():
            continue
        n = int(folder)
        if n > 99999:
            continue
        max_number = max(max_number, n)
    new_folder = os.path.join(base_path, f"{max_number + 1:03d}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

class AddBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        if self.training:
            return x + self.act(self.fc(x + torch.randn_like(x) * 0.001))
        else:
            return x + self.act(self.fc(x))
    
class SinCosEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        assert width % 2 == 0, "width msut be even"
        k = torch.arange(1, width // 2 + 1).view(1, -1).to(torch.float32) # [1, W/2]
        self.register_buffer("omega", math.pi * k) # pi * [1,2,...,W/2]

    def forward(self, x): # x: [B,1] and x\in[0,1]
        phase = x * self.omega # [B, W/2]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1) # [B, W]
    
class RGBEncoder(nn.Module):
    def __init__(self, width: int = 4, depth: int = 4, out_dim: int = 4):
        super().__init__()
        self.sin_width = (((width // 4) // 2) * 2)
        print(f"self.sin_width: {self.sin_width}")
        self.pre_proj = SinCosEmbedding(self.sin_width)
        
        self.in_proj = nn.Linear(self.sin_width * 4, width)
        self.in_act = nn.ReLU(inplace=False)
        self.blocks = nn.ModuleList([AddBlock(width) for _ in range(max(0, depth - 1))])
        self.out_proj = nn.Linear(width, out_dim, bias = False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.in_proj.bias)

    def forward(self, rgb):  # rgb: [B, 3]
        scale = torch.amax(rgb, dim=1, keepdim=True) # [B, 1]
        scale_eps = scale + 0.001
        rgb_sin = torch.cat([self.pre_proj(rgb[:, 0:1] / scale_eps), self.pre_proj(rgb[:, 1:2] / scale_eps), self.pre_proj(rgb[:, 2:3] / scale_eps), self.pre_proj(scale)], dim=-1)
        #h = self.in_act(self.in_proj(rgb ** 0.5))
        h = self.in_act(self.in_proj(rgb_sin))
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h) * scale

def clamp01(x):
    return F.relu(1.0 - F.relu(1.0 - x))
    
class LambdaEncoder(nn.Module):
    def __init__(self, in_dim: int = 1, width: int = 4, depth: int = 4, out_dim: int = 4):
        super().__init__()
        self.in_proj = SinCosEmbedding(width)
        self.blocks = nn.ModuleList([AddBlock(width) for _ in range(max(0, depth - 1))])
        self.out_proj = nn.Linear(width, out_dim, bias = False)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, lam):  # lam: [B, 1] (lambda_norm)
        h = self.in_proj(lam)
        for blk in self.blocks:
            h = blk(h)
        #return clamp01(self.out_proj(h))
        return self.out_proj(h)
    

class SpectralMLP(nn.Module):
    #Input:  [B, 4] = [r, g, b, lambda_norm]
    #Output: [B, 1]
    def __init__(self, width: int = 4, depth: int = 4, kernel: int = 4, affix_depth: int = 4):
        super().__init__()
        self.rgb_encoder = RGBEncoder(width=width, depth=depth, out_dim=kernel) # cache at host
        self.lambda_encoder = LambdaEncoder(in_dim=1, width=width, depth=affix_depth, out_dim=kernel) # cache per thread

    def forward(self, x):
        if self.training:
            x[0:x.shape[0]//2, 0:3] += torch.randn_like(x[0:x.shape[0]//2, 0:3]) * 0.025 / 256.0 # 
            x[0:x.shape[0]//2, 3:4] += torch.randn_like(x[0:x.shape[0]//2, 3:4]) * 0.025 / (780.0 - 360.0) # 
        # x: [B,4] -> [rgb(3), lambda_norm(1)]
        kernel_vec = self.rgb_encoder(x[:, 0:3]) # [B, K], cacheable
        mapped_vec = self.lambda_encoder(x[:, 3:4]) # [B, K], cacheable
        result = (kernel_vec * mapped_vec).sum(dim=-1, keepdim=True)
        #return F.sigmoid(result + self.interaction_encoder(x)) # [B, 1]
        return clamp01(result) # [B, 1]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_acc, mae_acc, n = 0.0, 0.0, 0
    mae_max = 0.0
    mse_max = 0.0
    for batch in loader:
        x = batch.to(device)
        rgb = x[:, 0:3]
        lam_nm = x[:, 3:4]
        val = x[:, 4:5]
        ds = loader.dataset
        lmin = getattr(ds, "lmin", 360.0)
        lmax = getattr(ds, "lmax", 780.0)
        lam_norm = (lam_nm - lmin) / (lmax - lmin)
        inp = torch.cat([rgb, lam_norm], dim=1)
        pred = model.forward(inp)
        mse = F.mse_loss(pred, val, reduction="sum").item()
        mae = F.l1_loss(pred, val, reduction="sum").item()
        bs = val.shape[0]
        mse_acc += mse
        mae_acc += mae
        mae_max = max(mae_max, mae / bs)
        mse_max = max(mse_max, mse / bs)
        n += bs
    model.train()
    if n == 0:
        return 0.0, 0.0
    return mse_acc / n, mse_max, mae_acc / n, mae_max


def _c_float_literal(v) -> str:
    s = f"{float(v):.9g}"
    if ("." not in s) and ("e" not in s) and ("E" not in s):
        s += ".0"
    return s + "f"

# output to cuda
def _fmt_array(name, arr, per_line=8, indent="    "):
    if arr.size == 0:
        return f""
    vals = ", ".join(f"{_c_float_literal(v)}" for v in arr.reshape(-1))
    chunks = []
    cur = []
    cnt = 0
    for x in vals.split(", "):
        cur.append(x)
        cnt += 1
        if cnt == per_line:
            chunks.append(indent + (", ".join(cur)) + ",")
            cur = []
            cnt = 0
    if cur:
        chunks.append(indent + (", ".join(cur)) + ",")
    body = "\n".join(chunks)
    gpu_str = f"alignas(16) static __device__ constexpr float {name}[{arr.size}] ="
    cpu_str = f"alignas(16) inline constexpr float {name}[{arr.size}] ="
    content = f"{{\n{body}\n}};"
    return f"#ifdef __CUDA_ARCH__\n{gpu_str}\n#else\n{cpu_str}\n#endif\n{content}\n"

def export_spectral_mlp_to_cuda(model, out_path: str,
                                width: int, depth: int, kernel: int, affix_depth: int,
                                lmin: float, lmax: float, mae_loss: float):
    sd = model.state_dict()
    def as_np(k):
        if k in sd:
            t = sd[k].detach().cpu().contiguous().numpy()
        else:
            t = np.array([])
        return t.astype(np.float32)

    # RGB
    rgb_in_w = as_np("rgb_encoder.in_proj.weight")    # [W,3]
    rgb_in_b = as_np("rgb_encoder.in_proj.bias")      # [W]
    rgb_blk_w = []
    rgb_blk_b = []
    for i in range(max(0, depth - 1)):
        rgb_blk_w.append(as_np(f"rgb_encoder.blocks.{i}.fc.weight"))  # [W,W]
        rgb_blk_b.append(as_np(f"rgb_encoder.blocks.{i}.fc.bias"))    # [W]
    rgb_out_w = as_np("rgb_encoder.out_proj.weight")  # [K,W]
    rgb_out_b = as_np("rgb_encoder.out_proj.bias")    # [K]

    # Lambda
    lam_in_w = as_np("lambda_encoder.in_proj.weight")   # [K,1]
    lam_in_b = as_np("lambda_encoder.in_proj.bias")     # [K]
    lam_blk_w = []
    lam_blk_b = []
    for i in range(max(0, affix_depth - 1)):
        lam_blk_w.append(as_np(f"lambda_encoder.blocks.{i}.fc.weight"))  # [K,K]
        lam_blk_b.append(as_np(f"lambda_encoder.blocks.{i}.fc.bias"))    # [K]
    lam_out_w = as_np("lambda_encoder.out_proj.weight") # [K,K]
    lam_out_b = as_np("lambda_encoder.out_proj.bias")   # [K]

    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f:
        f.write("namespace query\n")
        f.write("{\n")
        f.write(f"// mae: {mae_loss:.8f}\n")
        f.write(f"constexpr int WIDTH = {width};\n")
        f.write(f"constexpr int DEPTH = {depth};\n")
        f.write(f"constexpr int KERNEL = {kernel};\n")
        f.write(f"constexpr int AFFIX_DEPTH = {affix_depth};\n")
        f.write("static_assert((WIDTH % 4) == 0, \"WIDTH must be multiple of 4 for float4 loads\");\n")
        f.write("static_assert((KERNEL % 4) == 0, \"KERNEL must be multiple of 4 for float4 loads\");\n\n")

        # arrays
        f.write(_fmt_array("RGB_IN_W", rgb_in_w))
        f.write(_fmt_array("RGB_IN_B", rgb_in_b))
        for i,(w,b) in enumerate(zip(rgb_blk_w, rgb_blk_b)):
            f.write(_fmt_array(f"RGB_BLK{i}_W", w))
            f.write(_fmt_array(f"RGB_BLK{i}_B", b))
        f.write(_fmt_array("RGB_OUT_W", rgb_out_w))
        f.write(_fmt_array("RGB_OUT_B", rgb_out_b))

        f.write("\n")
        f.write(_fmt_array("LAM_IN_W", lam_in_w))
        f.write(_fmt_array("LAM_IN_B", lam_in_b))
        for i,(w,b) in enumerate(zip(lam_blk_w, lam_blk_b)):
            f.write(_fmt_array(f"LAM_BLK{i}_W", w))
            f.write(_fmt_array(f"LAM_BLK{i}_B", b))
        f.write(_fmt_array("LAM_OUT_W", lam_out_w))
        f.write(_fmt_array("LAM_OUT_B", lam_out_b))
        f.write(r"""
template<bool ReLU>
FUNCTION_MODIFIER_INLINE void AddBlockWidth(const float* __restrict__ xin, float* __restrict__ xout, const float* __restrict__ W, const float* __restrict__ B) 
{
    float tmp[WIDTH];
    tensor::LinearRowMajor<WIDTH, WIDTH, /*bBias=*/true, /*bReLU=*/true, /*bOpt=*/true>(xin, tmp, 0, 0, W, B);
    CUDA_UNROLL
    for (int i = 0; i < WIDTH; ++i) 
    {
        xout[i] = xin[i] + tmp[i];
    }
}

FUNCTION_MODIFIER_INLINE void RGBEncode(const float* __restrict__ in3, float* __restrict__ outK) 
{
    const float scale = fmaxf(fmaxf(in3[0], in3[1]), in3[2]);
    const float scaleEps = scale + 0.001f;
    const float scaleEpsInv = 1.0f / scaleEps;
    constexpr int RGB_SIN_WIDTH = WIDTH / 4 / 2 * 2;
    float s[RGB_SIN_WIDTH * 4];
    // sincos: s = [sin(piR), cos(piR), sin(2piR), cos(2piR), ..., sin(piG), cos(piG), sin(2piG), cos(2piG), ..., sin(piB), cos(piB), sin(2piB), cos(2piB)]
    {
        CUDA_UNROLL
        for (int k = 0; k < RGB_SIN_WIDTH/2; ++k)
        {
            const float phase = in3[0] * scaleEpsInv * CUDART_PI_F * static_cast<float>(k + 1);
            sincosf(phase, &s[k], &s[k + RGB_SIN_WIDTH/2]);
        }
        CUDA_UNROLL
        for (int k = 0; k < RGB_SIN_WIDTH/2; ++k)
        {
            const float phase = in3[1] * scaleEpsInv * CUDART_PI_F * static_cast<float>(k + 1);
            sincosf(phase, &s[k + RGB_SIN_WIDTH], &s[k + RGB_SIN_WIDTH/2 + RGB_SIN_WIDTH]);
        }
        CUDA_UNROLL
        for (int k = 0; k < RGB_SIN_WIDTH/2; ++k)
        {
            const float phase = in3[2] * scaleEpsInv * CUDART_PI_F * static_cast<float>(k + 1);
            sincosf(phase, &s[k + RGB_SIN_WIDTH * 2], &s[k + RGB_SIN_WIDTH/2 + RGB_SIN_WIDTH * 2]);
        }
        CUDA_UNROLL
        for (int k = 0; k < RGB_SIN_WIDTH/2; ++k)
        {
            const float phase = scale * CUDART_PI_F * static_cast<float>(k + 1);
            sincosf(phase, &s[k + RGB_SIN_WIDTH * 3], &s[k + RGB_SIN_WIDTH/2 + RGB_SIN_WIDTH * 3]);
        }
    }
    float h[WIDTH];
    tensor::LinearRowMajor<RGB_SIN_WIDTH * 4, WIDTH, /*bBias=*/true, /*ReLU=*/true, /*Opt=*/true>(s, h, 0, 0, RGB_IN_W, RGB_IN_B);
""")
        for i in range(max(0, depth - 1)):
            f.write(f"    {{\n        float t[WIDTH];\n        tensor::LinearRowMajor<WIDTH, WIDTH, /*bBias=*/true, /*bReLU=*/true, /*bOpt=*/true>(h, t, 0, 0, RGB_BLK{i}_W, RGB_BLK{i}_B); "
                    f"\n        CUDA_UNROLL\n        for (int ii=0; ii<WIDTH; ++ii)\n        {{\n            h[ii] = h[ii] + t[ii];\n        }}\n    }}\n")
        if rgb_out_b.size == 0:
            f.write("    tensor::LinearRowMajor<WIDTH, KERNEL, /*bBias=*/false, /*bReLU=*/false, /*Opt=*/true>(h, outK, 0, 0, RGB_OUT_W, nullptr);\n")
        else:
            f.write("    tensor::LinearRowMajor<WIDTH, KERNEL, /*bBias=*/true, /*bReLU=*/false, /*Opt=*/true>(h, outK, 0, 0, RGB_OUT_W, RGB_OUT_B);\n")
        f.write("    CUDA_UNROLL\n    for (int i = 0; i < KERNEL; ++i)\n    {\n        outK[i] *= scale;\n    }\n}\n")

        f.write(r"""
FUNCTION_MODIFIER_INLINE void LambdaEncode(const float* __restrict__ in1, float* __restrict__ outK) 
{
""")
        f.write(f"    const float x = fminf(fmaxf((in1[0] - {lmin}f) / ({lmax}f - {lmin}f), 0.0f), 1.0f);\n")
        f.write(r"""
    float h[WIDTH];
    // sincos: h = [sin(piR), cos(piR), sin(2piR), cos(2piR), ..., sin(piG), cos(piG), sin(2piG), cos(2piG), ..., sin(piB), cos(piB), sin(2piB), cos(2piB)]
    {
        //constexpr float TWO_PI = 6.28318530717958647692f;
        CUDA_UNROLL
        for (int k = 0; k < WIDTH/2; ++k)
        {
            const float phase = x * CUDART_PI_F * static_cast<float>(k + 1);
            sincosf(phase, &h[k], &h[k + WIDTH/2]);
        }
    }
""")
        for i in range(max(0, affix_depth - 1)):
            f.write(f"    {{\n        float t[WIDTH];\n        tensor::LinearRowMajor<WIDTH, WIDTH, /*bBias=*/true, /*bReLU=*/true, /*bOpt=*/true>(h, t, 0, 0, LAM_BLK{i}_W, LAM_BLK{i}_B); "
                    f"\n        CUDA_UNROLL\n        for (int ii=0; ii<WIDTH; ++ii)\n        {{\n            h[ii] = h[ii] + t[ii];\n        }}\n    }}\n")
        if lam_out_b.size == 0:
            f.write("    tensor::LinearRowMajor<WIDTH, KERNEL, /*bBias=*/false, /*bReLU=*/false, /*Opt=*/true>(h, outK, 0, 0, LAM_OUT_W, nullptr);\n}\n")
        else:
            f.write("    tensor::LinearRowMajor<WIDTH, KERNEL, /*bBias=*/true, /*bReLU=*/false, /*Opt=*/true>(h, outK, 0, 0, LAM_OUT_W, LAM_OUT_B);\n}\n")
        f.write(r"""    
FUNCTION_MODIFIER_INLINE float SpectralInteract(const float* __restrict__ RGBKernel, const float* __restrict__ LambdaKernel) 
{
    float acc = 0.0f;
    CUDA_UNROLL
    for (int i = 0; i < KERNEL; ++i) 
    {
        acc = fmaf(RGBKernel[i], LambdaKernel[i], acc);
    }
    return fminf(fmaxf(acc, 0.0f), 1.0f);
}
""")
        testN = 100
        rng = np.random.default_rng(2025)
        x4 = rng.random((testN, 4), dtype=np.float32)  # [testN,4], \in[0,1]
        with torch.no_grad():
            model.eval()
            x4_t = torch.from_numpy(x4).to(next(model.parameters()).device)
            gt = model.forward(x4_t).view(-1).cpu().numpy().astype(np.float32)

        def _fmt_array2d_input(name, a):  # a shape = (testN,4)
            rows = []
            for c in range(4):
                rows.append("    { " + ", ".join(_c_float_literal(a[i, c]) for i in range(testN)) + " },")
            body = "\n".join(rows)
            gpu_str = f"alignas(16) static __device__ constexpr float {name}[4][{testN}] ="
            cpu_str = f"alignas(16) inline constexpr float {name}[4][{testN}] ="
            content = f"{{\n{body}\n}};\n"
            return f"#ifdef __CUDA_ARCH__\n{gpu_str}\n#else\n{cpu_str}\n#endif\n{content}"

        def _fmt_array1d_ro(name, a):
            vals = ", ".join(_c_float_literal(v) for v in a.reshape(-1))
            gpu_str = f"alignas(16) static __device__ constexpr float {name}[{a.size}] ="
            cpu_str = f"alignas(16) inline constexpr float {name}[{a.size}] ="
            content = f"{{\n    {vals}\n}};\n"
            return f"#ifdef __CUDA_ARCH__\n{gpu_str}\n#else\n{cpu_str}\n#endif\n{content}"
        x4[:, -1] = x4[:, -1] * (lmax - lmin) + lmin
        f.write(_fmt_array2d_input("inputTest", x4))  # float inputTest[4][testN]
        f.write(_fmt_array1d_ro("gtTest", gt))        # float gt[testN]
        f.write(r"""
FUNCTION_MODIFIER_INLINE void SelfTestSimple()
{
    float mse = 0.0f;
""")
        f.write(f"    for (int i = 0; i < {testN}; ++i)")
        f.write(r"""
    {
        const float rgb3[3] = { inputTest[0][i], inputTest[1][i], inputTest[2][i] };
        const float lam1[1] = { inputTest[3][i] };

        float k1[KERNEL];
        float k2[KERNEL];

        RGBEncode(rgb3, k1);
        LambdaEncode(lam1, k2);
        const float pred = SpectralInteract(k1, k2);

        const float err = pred - gtTest[i];
        mse += err * err;

        std::printf("[test #%02d] pred=%.9f  gt=%.9f  err=%.9e\n", i, pred, gtTest[i], err);
    }
""")
        f.write(f"    mse /= {testN}.0f;")
        f.write(r"""    std::printf("[test] MSE=%.9e\n", mse);
}
""")
        f.write("}")
    print(f"[OK] wrote {out_path}")
    

def LambdaToCIE1931_X(lam):
    kernel1 = (lam - 595.8) / 33.33
    kernel2 = (lam - 446.8) / 19.44
    return 1.065 * torch.exp(-0.5 * kernel1 * kernel1) + 0.366 * torch.exp(-0.5 * kernel2 * kernel2)

def LambdaToCIE1931_Y(lam):
    LN_556_3 = 6.3213077
    kernel = (torch.log(lam) - LN_556_3) / 0.075
    return 1.014 * torch.exp(-0.5 * kernel * kernel)

def LambdaToCIE1931_Z(lam):
    LN_449_8 = 6.1088030
    kernel = (torch.log(lam) - LN_449_8) / 0.051
    return 1.839 * torch.exp(-0.5 * kernel * kernel)

def LambdaToXYZ(lam): # [B, 1]
    lam = lam * (780.0 - 360.0) + 360.0
    xyz = torch.cat([LambdaToCIE1931_X(lam), LambdaToCIE1931_Y(lam), LambdaToCIE1931_Z(lam)], dim = -1) # [B, 1] -> [B, 3]
    return xyz

def _align_linear_const0(sd, shape):
    return sd.copy().align(
        shape,
        interpolator=colour.LinearInterpolator,
        extrapolator=colour.Extrapolator,
        extrapolator_kwargs={"method": "Constant", "left": 0.0, "right": 0.0},
    )

# ---- deterministic 1 nm integration using the same target as your dataset (Mallett2019) ----
def integrate_rgb_via_dataset(rgb_linear, lmin=360.0, lmax=780.0, step=1.0):
    shape = SpectralShape(lmin, lmax, step)
    sd = colour.recovery.RGB_to_sd_Mallett2019(np.asarray(rgb_linear, dtype=np.float64))
    sd_i = _align_linear_const0(sd, shape)

    cmfs = _align_linear_const0(
        colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"], shape
    )
    d65  = _align_linear_const0(colour.SDS_ILLUMINANTS["D65"], shape)

    # arrays
    L     = np.asarray(sd_i.values, dtype=np.float64)                 # [N]
    L = np.clip(L, 0.0, 1.0)
    xbar  = np.asarray(cmfs.values[..., 0], dtype=np.float64)         # [N]
    ybar  = np.asarray(cmfs.values[..., 1], dtype=np.float64)
    zbar  = np.asarray(cmfs.values[..., 2], dtype=np.float64)
    E     = np.asarray(d65.values, dtype=np.float64)                  # [N]
    dl    = float(shape.interval)                                      # 1.0

    S = np.sum(E * ybar) * dl
    k = 1.0 / S
    E_norm = E * k

    # XYZ
    X = np.sum(L * E_norm * xbar) * dl
    Y = np.sum(L * E_norm * ybar) * dl
    Z = np.sum(L * E_norm * zbar) * dl
    XYZ = np.array([X, Y, Z], dtype=np.float64)

    # linear sRGB
    rgb_rec = colour.XYZ_to_RGB(
        XYZ,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB,
    )
    rgb_rec = np.clip(rgb_rec, 0.0, 1.0)
    return XYZ, rgb_rec


# ---- optional: integrate using your trained SpectralMLP over 1 nm grid ----
def integrate_rgb_via_model(model, rgb_linear, lmin=360.0, lmax=780.0, step=1.0, device="cpu"):
    shape = SpectralShape(lmin, lmax, step)
    lambdas = np.arange(lmin, lmax + 1e-9, step, dtype=np.float32)
    lam_norm = (lambdas - lmin) / (lmax - lmin)

    rgb_tile = np.tile(np.asarray(rgb_linear, dtype=np.float32)[None, :], (lambdas.shape[0], 1))
    x = np.concatenate([rgb_tile, lam_norm[:, None]], axis=1)  # [N,4]

    with torch.no_grad():
        model.eval()
        pred = model(torch.from_numpy(x).to(device)).view(-1).cpu().numpy().astype(np.float64)
        L = np.clip(pred, 0.0, 1.0)

    # CMF / D65
    cmfs = _align_linear_const0(
        colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"], shape
    )
    d65  = _align_linear_const0(colour.SDS_ILLUMINANTS["D65"], shape)

    xbar  = np.asarray(cmfs.values[..., 0], dtype=np.float64)
    ybar  = np.asarray(cmfs.values[..., 1], dtype=np.float64)
    zbar  = np.asarray(cmfs.values[..., 2], dtype=np.float64)
    E     = np.asarray(d65.values, dtype=np.float64)
    dl    = float(shape.interval)

    # D65:sum(E*ybar)*dl == 1
    S = np.sum(E * ybar) * dl
    k = 1.0 / S
    E_norm = E * k

    X = np.sum(L * E_norm * xbar) * dl
    Y = np.sum(L * E_norm * ybar) * dl
    Z = np.sum(L * E_norm * zbar) * dl
    XYZ = np.array([X, Y, Z], dtype=np.float64)

    rgb_rec = colour.XYZ_to_RGB(
        XYZ,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB,
    )
    rgb_rec = np.clip(rgb_rec, 0.0, 1.0)
    return XYZ, rgb_rec

def validate_primaries_dataset(model, device):
    primaries = [
        ("white", np.array([1.0, 1.0, 1.0])),
        ("red",   np.array([1.0, 0.0, 0.0])),
        ("green", np.array([0.0, 1.0, 0.0])),
        ("blue",  np.array([0.0, 0.0, 1.0])),
    ]
    print("----------gt--------------")
    for name, rgb in primaries:
        XYZ, rgb_rec = integrate_rgb_via_dataset(rgb, lmin=360.0, lmax=780.0, step=1.0)
        XYZ_model, rgb_rec_model = integrate_rgb_via_model(model, rgb, lmin=360.0, lmax=780.0, step=1.0, device=device)
        print("-----------------------------")
        print(name)
        print(f"color   :   {{{rgb[0]:.8f}, {rgb[1]:.8f}, {rgb[2]:.8f}}}")
        print(f"xyz     :   {{{XYZ[0]:.8f}, {XYZ[1]:.8f}, {XYZ[2]:.8f}}}")
        print(f"xyz_pred:   {{{XYZ_model[0]:.8f}, {XYZ_model[1]:.8f}, {XYZ_model[2]:.8f}}}")
        print(f"rgb     :   {{{rgb_rec[0]:.8f}, {rgb_rec[1]:.8f}, {rgb_rec[2]:.8f}}}")
        print(f"rgb_pred:   {{{rgb_rec_model[0]:.8f}, {rgb_rec_model[1]:.8f}, {rgb_rec_model[2]:.8f}}}")
        
LAMBDA_MIN = 360.0
LAMBDA_MAX = 780.0
# main
def main(args):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(21 / 24, device=0)
        except Exception:
            pass

    assert args.config_path != "", "Missing config_path"
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # hparams from yaml
    width = int(config.get("model", {}).get("width", 4))
    depth = int(config.get("model", {}).get("depth", 2))
    kernel = int(config.get("model", {}).get("kernel", 4))
    affix_depth = int(config.get("model", {}).get("affix_depth", 2))
    lr = float(config.get("optim", {}).get("lr", 1e-3))
    weight_decay = float(config.get("optim", {}).get("weight_decay", 1e-4))
    grad_clip_norm = float(config.get("optim", {}).get("grad_clip_norm", 1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.global_seed)

    result_dir = get_new_folder_path(args.results_dir)
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # train/val datasets
    train_ds = SpectralQueryDataset(
        length=args.train_length,
        lambda_bounds=(LAMBDA_MIN, LAMBDA_MAX),
        lambda_step=args.lambda_step,
        rng=np.random.default_rng(args.global_seed),
    )
    val_ds = SpectralQueryDataset(
        length=args.val_length,
        lambda_bounds=(LAMBDA_MIN, LAMBDA_MAX),
        lambda_step=args.lambda_step,
        rng=np.random.default_rng(args.global_seed + 1),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size*10,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    print(f"width: {width}, depth: {depth}, kernel: {kernel}, affix_depth: {affix_depth}")
    model = SpectralMLP(width=width, depth=depth, kernel=kernel, affix_depth=affix_depth).to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = build_optimizer(model, "adamw", lr=lr, weight_decay=weight_decay)
    
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epoch * steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=0.001, min_lr_ratio=lr*0.01)
    
    ema = ModelEMA(model, decay=0.999)
    # resume
    start_step = 0
    if args.restore_path and os.path.isfile(args.restore_path):
        ckpt = torch.load(args.restore_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Restored from {args.restore_path} at step {start_step}")

    # training state
    model.train()
    iteration = start_step
    running_loss = 0.0
    log_steps = 0
    start_time = time()
    lmin = train_ds.lmin
    lmax = train_ds.lmax

    # best tracking
    best_score = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pth")
    best_export_path = os.path.join(ckpt_dir, "best.txt")

    def save_ckpt(path):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": iteration,
                "config": {
                    "model": {"width": width, "depth": depth},
                    "optim": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_clip_norm": grad_clip_norm,
                    },
                    "lambda_min": LAMBDA_MIN,
                    "lambda_max": LAMBDA_MAX,
                    "lambda_step": args.lambda_step,
                },
            },
            path,
        )

    for epoch in range(args.epoch):
        for batch in train_loader:
            x = batch.to(device)  # [B,5]
            rgb = x[:, 0:3]
            lam_nm = x[:, 3:4]
            val = x[:, 4:5]

            lam_norm = (lam_nm - lmin) / (lmax - lmin)
            inp = torch.cat([rgb, lam_norm], dim=1)  # [B,4]

            pred = model.forward(inp) # , math.sin(float(iteration) * 0.0007325) * 0.5 + 0.5
            assert pred.shape[1] == 1
            assert lam_norm.shape[1] == 1
            importance = 0.25 + torch.clamp(LambdaToXYZ(lam_norm), min = 0.0) # [B, 3]
            assert importance.shape[1] == 3
            loss = torch.square((pred - val) * importance).mean() # + torch.abs(pred - val).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            ema.update(model)
            scheduler.step()

            running_loss += loss.item()
            log_steps += 1
            iteration += 1

            # log
            if iteration % args.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time()
                steps_per_sec = log_steps / max(1e-6, (now - start_time))
                avg_loss = running_loss / max(1, log_steps)
                print(f"(step={iteration:07d}) loss={avg_loss:.6f} steps/s={steps_per_sec:.2f}")
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # eval + save best
            if iteration % args.eval_every == 0 or iteration == 1:
                mse, mse_max, mae, mae_max = evaluate(model, val_loader, device)
                mse_ema, mse_ema_max, mae_ema, mae_ema_max = evaluate(ema.ema, val_loader, device)
                validate_primaries_dataset(model, device)
                #
                score = mse_max # mae * 0.25 + mae_max * 0.75
                score_ema = mse_ema_max # mae_ema * 0.25 + mae_ema_max * 0.75
                print(f"[eval@{iteration:07d}] mse={mse:.6e} mae={mae:.6e}")
                print(f"[eval_ema@{iteration:07d}] mse={mse_ema:.6e} mae={mae_ema:.6e}")
                if score < best_score:
                    best_score = score
                    save_ckpt(best_path)
                    print(f"best checkpoint updated (score={best_score:.6e}), mse={mse:.6e}, mse_max={mse_max:.6e}, mae={mae:.6e}, mae_max={mae_max:.6e}: {best_path}")
                    export_spectral_mlp_to_cuda(model, best_export_path, width=width, depth=depth, kernel=kernel, affix_depth=affix_depth, lmin=lmin, lmax=lmax, mae_loss=mae_max)
                if score_ema < best_score:
                    best_score = score_ema
                    save_ckpt(best_path)
                    print(f"best checkpoint updated (score={best_score:.6e}), mse_ema={mse_ema:.6e}, mse_ema_max={mse_ema_max:.6e}, mae={mae_ema:.6e}, mae_max={mae_ema_max:.6e}: {best_path}")
                    export_spectral_mlp_to_cuda(ema.ema, best_export_path, width=width, depth=depth, kernel=kernel, affix_depth=affix_depth, lmin=lmin, lmax=lmax, mae_loss=mae_ema_max)

            # periodic save
            if iteration % args.ckpt_every == 0:
                path = os.path.join(ckpt_dir, f"{iteration}.pth")
                save_ckpt(path)
                print(f"checkpoint saved: {path}")

    # final save (regular)
    final_path = os.path.join(ckpt_dir, "final.pth")
    save_ckpt(final_path)
    print(f"final checkpoint saved: {final_path}")

    # also save minimal-loss model alongside final (copy or re-save)
    if os.path.exists(best_path):
        # optional: also drop a copy at results root for convenience
        best_copy = os.path.join(result_dir, "best.pth")
        if best_copy != best_path:
            import shutil
            shutil.copy2(best_path, best_copy)
            print(f"best checkpoint copied to: {best_copy}")
    else:
        # if no eval happened, save current as best
        save_ckpt(best_path)
        print(f"best checkpoint saved (no eval run): {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train-length", type=int, default=2500000)
    parser.add_argument("--val-length", type=int, default=2500000)
    #parser.add_argument("--val-length", type=int, default=500000)
    parser.add_argument("--lambda-step", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)

    # yaml config
    parser.add_argument("--config-path", type=str, default="./config.ini")

    # train
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=2000)

    # io
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--restore-path", type=str, default="")

    args = parser.parse_args()
    main(args)