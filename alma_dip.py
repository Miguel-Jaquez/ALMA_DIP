# alma_dip.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Device helper
# ----------------------------
def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.backends.mps.is_available():       # Apple GPU
        return torch.device("mps")
    if torch.cuda.is_available():               # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Config
# ----------------------------
@dataclass
class DIPConfig:
    seed: int = 123
    num_iters: int = 2500
    lr: float = 1e-3
    tv_weight: float = 5e-5
    input_depth: int = 32
    base_channels: int = 64
    depth: int = 5
    dropout: float = 0.0
    positivity: bool = True
    cell_size_arcsec: float = 0.10            # image pixel size
    nu: float = 3.0                           # degrees of freedom for Student's-t
    learn_sigma: bool = False                 # learn global scale in Student's-t
    init_sigma: float = 1.0
    out_every: int = 200                      # print every N iters


# ----------------------------
# TV loss (fixed shape bug)
# ----------------------------
def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Total-variation loss that does not mix dx & dy shapes.
    Works for tensors of shape [B, C, H, W].
    """
    dx = x[..., :, 1:] - x[..., :, :-1]      # [B,C,H, W-1]
    dy = x[..., 1:, :] - x[..., :-1, :]      # [B,C,H-1,W]
    return torch.sum(torch.sqrt(dx * dx + eps)) + torch.sum(torch.sqrt(dy * dy + eps))


# ----------------------------
# Student's-t negative log-likelihood
# ----------------------------
class StudentTLoss(nn.Module):
    """
    Robust NLL in visibility space. Each residual is 2D (Re, Im).
    We treat per-visibility 'weight' as a precision (1/variance). If 'learn_sigma'
    is enabled, a global scale sigma is learned as well (softplus-parametrized).
    """
    def __init__(self, nu: float = 3.0, learn_sigma: bool = False,
                 init_sigma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.nu = float(nu)
        self.reduction = reduction
        self.learn_sigma = learn_sigma
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma))))
        else:
            # kept as buffer so it moves with .to(device)
            self.register_buffer("log_sigma", torch.log(torch.tensor(float(init_sigma))))

        # dimensionality of residual (Re, Im)
        self.d = 2
        self.eps = 1e-8

    def forward(self,
                pred: torch.Tensor,    # [N, 2]
                target: torch.Tensor,  # [N, 2]
                weight: Optional[torch.Tensor] = None  # [N]
                ) -> torch.Tensor:
        r = pred - target                            # [N,2]
        r2 = torch.sum(r * r, dim=-1)               # [N], squared norm in R^2
        if weight is None:
            weight = torch.ones_like(r2)
        sigma2 = torch.exp(2.0 * self.log_sigma)    # global scale^2 (if learn_sigma)
        # Heteroscedastic scaling: weight multiplies squared residual
        scaled = 1.0 + (weight * r2) / (self.nu * sigma2 + self.eps)
        nll = 0.5 * (self.nu + self.d) * torch.log(scaled + self.eps)
        # include sigma-dependence term if sigma is learnable
        if self.learn_sigma:
            nll = nll + 0.5 * self.d * self.log_sigma
        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        return nll


# ----------------------------
# DIP generator (robust U-Net)
#  - reflection padding
#  - group norm (BN can be temperamental on small batches)
#  - bilinear upsampling (no transpose conv checkerboards)
# ----------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        # Use 8 groups or 1 if channels < 8
        groups = max(1, min(8, out_ch))
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvGNAct(in_ch + skip_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if shapes differ by 1 due to odd dimensions
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DIPUNet(nn.Module):
    def __init__(self, input_depth: int = 32, base_ch: int = 64, depth: int = 5,
                 dropout: float = 0.0, out_ch: int = 1):
        super().__init__()
        self.depth = depth
        chs = [base_ch * min(2 ** i, 8) for i in range(depth)]  # cap growth
        self.downs = nn.ModuleList()
        in_c = input_depth
        for i in range(depth):
            out_c = chs[i]
            self.downs.append(Down(in_c, out_c, dropout))
            in_c = out_c

        self.bottleneck = nn.Sequential(
            ConvGNAct(chs[-1], chs[-1], dropout),
            ConvGNAct(chs[-1], chs[-1], dropout),
        )

        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            in_c = chs[i]
            skip_c = chs[i]
            out_c = chs[i - 1] if i > 0 else base_ch
            self.ups.append(Up(in_c, skip_c, out_c, dropout))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=0)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skips = []
        x = z
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x, skips[-(i + 1)])
        x = self.final(x)
        return x


# ----------------------------
# NUDFT helper (MPS-friendly; real trig only)
# ----------------------------
@torch.no_grad()
def _grid_coords(H: int, W: int, cell_size_rad: float, device: torch.device, dtype: torch.dtype):
    l = (torch.arange(W, device=device, dtype=dtype) - (W // 2)) * cell_size_rad  # [W]
    m = (torch.arange(H, device=device, dtype=dtype) - (H // 2)) * cell_size_rad  # [H]
    return l, m


def precompute_uv_phases(uv: torch.Tensor, H: int, W: int, cell_size_rad: float,
                         device: torch.device, dtype: torch.dtype = torch.float32):
    """
    Precompute cos/sin for exp(±i 2π (u l + v m)) using separability.
    Returns Cul, Sul (N,W) and Cvm, Svm (N,H).
    """
    u = uv[:, 0].to(device=device, dtype=dtype)  # [N]
    v = uv[:, 1].to(device=device, dtype=dtype)  # [N]
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)
    ul = 2.0 * math.pi * (u[:, None] * l[None, :])  # [N,W]
    vm = 2.0 * math.pi * (v[:, None] * m[None, :])  # [N,H]
    Cul = torch.cos(ul);  Sul = torch.sin(ul)       # [N,W]
    Cvm = torch.cos(vm);  Svm = torch.sin(vm)       # [N,H]
    return Cul, Sul, Cvm, Svm


def predict_vis_from_precomp(img: torch.Tensor,
                             Cul: torch.Tensor, Sul: torch.Tensor,
                             Cvm: torch.Tensor, Svm: torch.Tensor) -> torch.Tensor:
    """
    img: [1,1,H,W], real sky brightness.
    Returns pred visibilities as [N,2] (Re, Im) using exp(-iθ) convention.
    """
    I = img[0, 0]                         # [H,W]
    # A = I @ Cul^T, B = I @ Sul^T   => [H,N]
    A = I @ Cul.t()                       # [H,N]
    B = I @ Sul.t()                       # [H,N]
    # Real = sum_h Cvm[:,h]*A[h,:]  - sum_h Svm[:,h]*B[h,:]
    real = torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)  # [N]
    # Imag = -[ sum_h Cvm[:,h]*B[h,:] + sum_h Svm[:,h]*A[h,:] ]
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))  # [N]
    return torch.stack([real, imag], dim=1)  # [N,2]


@torch.no_grad()
def make_dirty_and_psf(Cul: torch.Tensor, Sul: torch.Tensor,
                       Cvm: torch.Tensor, Svm: torch.Tensor,
                       re: torch.Tensor, im: torch.Tensor, w: torch.Tensor,
                       normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized synthesis of dirty image and PSF using exp(+iθ).
    re, im, w: [N]
    Returns (dirty, psf) each [H,W].
    """
    H = Cvm.shape[1]
    W = Cul.shape[1]
    # E = w*a*Cul - w*b*Sul,   F = w*a*Sul + w*b*Cul
    E = (w * re)[:, None] * Cul - (w * im)[:, None] * Sul          # [N,W]
    F = (w * re)[:, None] * Sul + (w * im)[:, None] * Cul          # [N,W]
    # Dirty = Cvm^T @ E - Svm^T @ F
    dirty = Cvm.t() @ E - Svm.t() @ F                              # [H,W]
    # PSF: set a=1, b=0  => Epsf = w*Cul,  Fpsf = w*Sul
    Epsf = w[:, None] * Cul
    Fpsf = w[:, None] * Sul
    psf = Cvm.t() @ Epsf - Svm.t() @ Fpsf                          # [H,W]
    if normalize:
        norm = torch.sum(w).clamp(min=1e-12)
        dirty = dirty / norm
        psf = psf / norm
    return dirty, psf


# ----------------------------
# High-level reconstruction
# ----------------------------
def reconstruct_dip(
    uv: np.ndarray | torch.Tensor,
    vis: Optional[np.ndarray | torch.Tensor] = None,
    weight: Optional[np.ndarray | torch.Tensor] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args
    ----
    uv:     [N,2] array/tensor of (u, v) in wavelengths,
            OR [N,5] with columns (u, v, Re, Im, weight) if 'vis' and 'weight' are None.
    vis:    [N,2] (Re, Im) in Jy (optional if uv already has 5 cols)
    weight: [N]   weights (≈ 1/σ^2) per visibility
    img_size: (H, W) output image size
    cfg:    DIPConfig
    device: 'mps' | 'cuda' | 'cpu' or None for auto

    Returns
    -------
    (image, dirty, beam) as numpy arrays of shape [H,W].
    """
    if cfg is None:
        cfg = DIPConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dev = pick_device(device)

    # Parse inputs
    def to_t(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2:
        raise ValueError("uv must be 2D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone()
        im_t = uv_t[:, 3].clone()
        w_t = uv_t[:, 4].clone()
        uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None:
            raise ValueError("Either provide vis & weight, or pass uv with 5 columns.")
        v_t = to_t(vis)
        if v_t.shape[1] != 2:
            raise ValueError("vis must have shape [N,2] (Re, Im)")
        re_t = v_t[:, 0].clone()
        im_t = v_t[:, 1].clone()
        w_t = to_t(weight).clone()

    # Basic shapes & constants
    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648000.0)  # arcsec -> rad

    # Precompute trig tables (MPS-friendly)
    Cul, Sul, Cvm, Svm = precompute_uv_phases(uv_t, H, W, cell_size_rad, dev, torch.float32)

    # Dirty image & PSF (for reference / return)
    with torch.no_grad():
        dirty_t, psf_t = make_dirty_and_psf(Cul, Sul, Cvm, Svm, re_t, im_t, w_t, normalize=True)

    # Build DIP net
    net = DIPUNet(
        input_depth=cfg.input_depth,
        base_ch=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
        out_ch=1,
    ).to(dev)

    # Fixed noise input
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    # Positivity transform (softplus) to keep the sky nonnegative
    def pos(x):
        return F.softplus(x) if cfg.positivity else x

    # Loss
    student = StudentTLoss(nu=cfg.nu, learn_sigma=cfg.learn_sigma,
                           init_sigma=cfg.init_sigma, reduction="mean").to(dev)

    optimizer = torch.optim.Adam(list(net.parameters()) + (
        [student.log_sigma] if cfg.learn_sigma else []), lr=cfg.lr)

    target_vis = torch.stack([re_t, im_t], dim=1)      # [N,2]
    weights = torch.clamp(w_t, min=1e-12)              # [N]

    best_data = float("inf")
    best_img = None

    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))                               # [1,1,H,W]

        pred_vis = predict_vis_from_precomp(img, Cul, Sul, Cvm, Svm)  # [N,2]

        data_term = student(pred_vis, target_vis, weight=weights)
        reg = cfg.tv_weight * tv_loss(img)

        loss = data_term + reg
        loss.backward()
        optimizer.step()

        if (it + 1) % cfg.out_every == 0 or it == 0:
            with torch.no_grad():
                data_val = float(data_term.detach().cpu())
                tv_val = float(reg.detach().cpu())
            print(f"[{it+1:5d}/{cfg.num_iters}]  data(t)={data_val:.6f}  tv={tv_val:.6f}"
                  + (f"  sigma={float(torch.exp(student.log_sigma).detach().cpu()):.4g}" if cfg.learn_sigma else ""))

        # Track best (by data fidelity)
        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item()
                best_img = img.detach().clone()

    # Fallback to last if best didn't set (unlikely)
    if best_img is None:
        best_img = img.detach()

    # Return numpy arrays
    image = best_img[0, 0].detach().cpu().numpy()
    dirty = dirty_t.detach().cpu().numpy()
    beam = psf_t.detach().cpu().numpy()
    return image, dirty, beam
