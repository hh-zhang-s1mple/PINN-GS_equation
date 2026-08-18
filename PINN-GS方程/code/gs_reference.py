# -*- coding: utf-8 -*-
"""gs_reference.py —— 有限差分(FD)参考解：广义特征值问题

背景：
  源项 p'(ψ)=ALPHA·ψ, (f²)'(ψ)=BETA·ψ 时，GS 方程
      Δ*ψ = -μ0 R² p'(ψ) - 0.5 (f²)'(ψ),   ψ|边界 = 0
  化为线性齐次问题  -Δ*ψ = c(R)·ψ,  c(R) = μ0 R² ALPHA + BETA/2。
  这是广义特征值问题 -Δ*ψ = λ·c(R)·ψ：仅当 λ₁ = 1（基模）时方程才有
  满足"磁轴 ψ=1"的非平凡解；λ₁ ≠ 1 时只有零解，PINN 只能得到折中解。

用法：
  python gs_reference.py            # ALPHA=BETA=1.0：输出 λ₁、基模 ψ_ref、q(ψ)
  python gs_reference.py 2.13 1.0   # 指定 ALPHA, BETA
  python gs_reference.py scan       # 固定 BETA=1，扫描 ALPHA 找 λ₁=1 的自洽值
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows GBK 控制台打印 λ/ψ 兼容
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

R_MIN, R_MAX, Z_MIN, Z_MAX = 1.2, 2.8, -0.8, 0.8
R0, A = 2.0, 0.8
MU0 = 1.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 网格与算子装配 ============


def build_operator(nr, nz):
    R = np.linspace(R_MIN, R_MAX, nr)
    Z = np.linspace(Z_MIN, Z_MAX, nz)
    Rg, Zg = np.meshgrid(R, Z, indexing="ij")
    inside = (Rg - R0) ** 2 + Zg ** 2 < A ** 2          # 圆内点；圆上/圆外 ψ=0
    idx = np.full((nr, nz), -1, dtype=int)
    idx[inside] = np.arange(inside.sum())
    n = inside.sum()
    h = (R_MAX - R_MIN) / (nr - 1)
    hz = (Z_MAX - Z_MIN) / (nz - 1)
    S = np.zeros((n, n))                                # -Δ* 的离散矩阵
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            p = idx[i, j]
            if p < 0:
                continue
            stencil = np.zeros((nr, nz))
            stencil[i, j] = 2.0 / h ** 2 + 2.0 / hz ** 2
            stencil[i + 1, j] = -1.0 / h ** 2 + 1.0 / (2.0 * h * Rg[i, j])
            stencil[i - 1, j] = -1.0 / h ** 2 - 1.0 / (2.0 * h * Rg[i, j])
            stencil[i, j + 1] = -1.0 / hz ** 2
            stencil[i, j - 1] = -1.0 / hz ** 2
            S[p, :] = stencil[inside]
    return R, Z, Rg, Zg, inside, S


def solve_fundamental(alpha, beta, nr=61, nz=61):
    """求最小广义特征值 λ₁ 与基模 ψ（已归一化 ψ(轴)=1）。"""
    R, Z, Rg, Zg, inside, S = build_operator(nr, nz)
    c = MU0 * Rg[inside] ** 2 * alpha + 0.5 * beta        # c(R) = μ0 R²α + β/2
    S_t = torch.from_numpy(S).double().to(DEV)
    D_inv = torch.diag(1.0 / torch.from_numpy(c).double().to(DEV))
    evals, evecs = torch.linalg.eig(D_inv @ S_t)
    lam = evals.real.cpu().numpy()
    good = (np.abs(evals.imag.cpu().numpy()) < 1e-6 * np.abs(lam) + 1e-8) & (lam > 0) & (lam < 1e3)
    order = np.argsort(lam[good])
    lam1 = lam[good][order[0]]
    v = evecs[:, np.where(good)[0][order[0]]].real.cpu().numpy()
    i_ax = np.argmin(np.abs(Rg[inside] - R0) + np.abs(Zg[inside]))
    v = v / v[i_ax]                                       # 归一化：ψ(磁轴)=1（带符号，保证整体为正）
    psi = np.zeros((nr, nz))
    psi[inside] = v
    return R, Z, Rg, Zg, psi, lam1


# ============ 双线性插值（numpy 手写，不依赖 scipy） ============


def bilinear(field, R, Z, rq, zq):
    dr = R[1] - R[0]
    dz = Z[1] - Z[0]
    fr = (rq - R[0]) / dr
    fz = (zq - Z[0]) / dz
    i = np.clip(fr.astype(int), 0, len(R) - 2)
    j = np.clip(fz.astype(int), 0, len(Z) - 2)
    wr = fr - i
    wz = fz - j
    return (field[i, j] * (1 - wr) * (1 - wz) + field[i + 1, j] * wr * (1 - wz)
            + field[i, j + 1] * (1 - wr) * wz + field[i + 1, j + 1] * wr * wz)


def fine_grid(R, Z, psi, nr_f=241, nz_f=241):
    Rf = np.linspace(R_MIN, R_MAX, nr_f)
    Zf = np.linspace(Z_MIN, Z_MAX, nz_f)
    Rfg, Zfg = np.meshgrid(Rf, Zf, indexing="ij")
    tmp = np.zeros((nr_f, len(Z)))
    for j in range(len(Z)):
        tmp[:, j] = np.interp(Rf, R, psi[:, j])
    psi_f = np.zeros((nr_f, nz_f))
    for i in range(nr_f):
        psi_f[i, :] = np.interp(Zf, Z, tmp[i, :])
    return Rf, Zf, Rfg, Zfg, psi_f


# ============ q(ψ)：极向线积分（matplotlib.contour + numpy 梯形法则） ============


def q_profile(Rg, Zg, psi, beta, levels):
    """q(ψ) = F(ψ)/(2π) ∮ dl/(R²|∇ψ|)，F = sqrt(f0² + β/2·(ψ²-1))，f0=1。"""
    dR = Rg[1, 0] - Rg[0, 0]
    dZ = Zg[0, 1] - Zg[0, 0]
    gR, gZ = np.gradient(psi, dR, dZ)
    fig = plt.figure()
    cs = plt.contour(Rg, Zg, psi, levels=levels)
    qs = []
    for k, segs in enumerate(cs.allsegs):
        if len(segs) == 0:
            qs.append(np.nan)
            continue
        total = 0.0
        for seg in segs:
            if np.hypot(seg[0, 0] - seg[-1, 0], seg[0, 1] - seg[-1, 1]) > 1e-12:
                seg = np.vstack([seg, seg[:1]])           # 确保路径闭合
            gRs = bilinear(gR, Rg[:, 0], Zg[0, :], seg[:, 0], seg[:, 1])
            gZs = bilinear(gZ, Rg[:, 0], Zg[0, :], seg[:, 0], seg[:, 1])
            grad_mag = np.hypot(gRs, gZs)
            integ = 1.0 / (seg[:, 0] ** 2 * grad_mag + 1e-10)
            dseg = np.diff(seg, axis=0)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(dseg[:, 0], dseg[:, 1]))])
            total += np.trapezoid(integ, arc)             # numpy 梯形法则(np.trapz 的现名)
        F = np.sqrt(1.0 + beta / 2.0 * (levels[k] ** 2 - 1.0))
        qs.append(F / (2.0 * np.pi) * total)
    plt.close(fig)
    return np.array(qs)


# ============ 扫描模式：固定 BETA，找 λ₁=1 的 ALPHA ============


def scan_alpha(beta=1.0):
    alphas = np.arange(1.0, 3.05, 0.1)
    lams = np.array([solve_fundamental(a, beta, 41, 41)[5] for a in alphas])
    k = np.where(lams <= 1.0)[0]
    if len(k) == 0:
        raise SystemExit("扫描范围内未找到 λ₁=1，请扩大范围")
    lo, hi = alphas[k[0] - 1], alphas[k[0]]
    alphas_r = np.arange(lo, hi + 1e-12, 0.01)
    lams_r = np.array([solve_fundamental(a, beta, 41, 41)[5] for a in alphas_r])
    i_star = np.argmin(np.abs(lams_r - 1.0))
    np.savez("figures/lambda_scan.npz", alphas=np.concatenate([alphas, alphas_r]),
             lams=np.concatenate([lams, lams_r]), alpha_star=alphas_r[i_star], beta=beta)
    return alphas_r[i_star], lams_r[i_star]


# ============ 主流程 ============


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        a_star, l_star = scan_alpha(1.0)
        print(f"[扫描] 固定 BETA=1.0，λ₁=1 的自洽 ALPHA* = {a_star:.4f} (λ₁={l_star:.4f})")
        alpha, beta = float(a_star), 1.0
    else:
        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
        beta = float(sys.argv[2]) if len(sys.argv) > 2 else alpha

    R, Z, Rg, Zg, psi, lam1 = solve_fundamental(alpha, beta, 61, 61)
    i_ax = np.argmin(np.abs(Rg - R0) + np.abs(Zg))
    iax, jax = np.unravel_index(i_ax, Rg.shape)
    imax, jmax = np.unravel_index(np.argmax(psi), Rg.shape)
    print("=" * 56)
    print(f"[FD 参考解]  ALPHA={alpha}, BETA={beta}, 网格 61x61")
    print(f"  基模广义特征值 λ₁ = {lam1:.4f}   (自洽要求: λ₁ = 1)")
    print(f"  ψ(磁轴)= {psi[iax, jax]:.4f}   全域最大 ψ = {psi[imax, jmax]:.4f}"
          f" @ (R={Rg[imax, jmax]:.3f}, Z={Zg[imax, jmax]:.3f})")
    if abs(lam1 - 1.0) > 0.02:
        print(f"  >> λ₁≠1：α={alpha},β={beta} 下方程无精确解（只有零解），"
              f"PINN 只能得到折中解。")
        print(f"  >> 提示：保持 BETA={beta} 时可用 `python gs_reference.py scan` 找自洽 ALPHA。")
    else:
        print("  >> λ₁≈1：方程自洽，存在满足磁轴约束的非平凡解。")
    levels = np.linspace(0.1, 0.9, 9)
    qs = q_profile(Rg, Zg, psi, beta, levels)
    print("  q(ψ) 参考值:")
    for lv, q in zip(levels, qs):
        print(f"    ψ={lv:.2f}  q={q:.4f}")

    Rf, Zf, Rfg, Zfg, psi_f = fine_grid(R, Z, psi)
    np.savez("figures/psi_ref.npz", R=R, Z=Z, Rg=Rg, Zg=Zg, psi=psi,
             Rf=Rf, Zf=Zf, Rfg=Rfg, Zfg=Zfg, psi_f=psi_f,
             levels=levels, q=qs, lam1=lam1, alpha=alpha, beta=beta)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cf = ax.contourf(Rfg, Zfg, psi_f, levels=24, cmap="viridis")
    ax.contour(Rfg, Zfg, psi_f, levels=24, colors="white", linewidths=0.4)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R0 + A * np.cos(th), A * np.sin(th), "r--", lw=1.6, label=r"边界 $\psi=0$")
    ax.plot([R0], [0.0], "r*", ms=14, label=r"磁轴 $\psi=1$")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"FD 参考解 (α={alpha}, β={beta}, $\\lambda_1$={lam1:.4f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("scaled")
    fig.colorbar(cf, label=r"$\psi$")
    fig.tight_layout()
    fig.savefig("figures/ref_psi.png", dpi=150)
    plt.close(fig)
    print("  已保存: figures/psi_ref.npz, figures/ref_psi.png")


if __name__ == "__main__":
    main()
