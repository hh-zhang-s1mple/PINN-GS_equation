# -*- coding: utf-8 -*-
"""make_doc_figures.py 
用法：python make_doc_figures.py
      （含 α=β=1 的原始运行与 α=2.0,β=1.0 的修复运行，后者文件名带 _consistent 后缀）
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("doc_figures", exist_ok=True)


def parse_log(path):
    raw = open(path, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    eps, losses = [], []
    for line in text.splitlines():
        m = re.search(r"epoch\s+(\d+)\s+loss=([0-9.eE+-]+)", line)
        if m:
            eps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return np.array(eps), np.array(losses)


def bilinear(field, R, Z, rq, zq):
    dr = R[1] - R[0]
    dz = Z[1] - Z[0]
    fr = np.clip((rq - R[0]) / dr, 0, len(R) - 2)
    fz = np.clip((zq - Z[0]) / dz, 0, len(Z) - 2)
    i = fr.astype(int)
    j = fz.astype(int)
    wr = fr - i
    wz = fz - j
    return (field[i, j] * (1 - wr) * (1 - wz) + field[i + 1, j] * wr * (1 - wz)
            + field[i, j + 1] * (1 - wr) * wz + field[i + 1, j + 1] * wr * wz)


# ---------- 图 1：损失曲线 ----------
e1, l1 = parse_log("training_log_phase1.txt")
e2a, l2a = parse_log("training_log_phase2_ab11.txt")
e2c, l2c = parse_log("training_log_phase2_consistent.txt")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(e1, l1, "o-", ms=3, lw=1.2)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("总损失")
ax[0].set_title("阶段一（500 步验证导数）")
ax[0].grid(alpha=0.3)
ax[1].semilogy(e2a, l2a, "o-", ms=3, lw=1.2, label="α=β=1（无精确解 → 折中地板）")
ax[1].semilogy(e2c, l2c, "s-", ms=3, lw=1.2, label="α=2.0, β=1.0（自洽 → 持续下降）")
ax[1].axhline(l2a[-1], color="tab:blue", ls="--", lw=1, alpha=0.6, label=f"地板值 {l2a[-1]:.3f}")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("总损失（对数轴）")
ax[1].set_title("阶段二（5000 步）：同一套代码，只有 ALPHA 不同")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("doc_figures/loss_curves.png", dpi=150)
plt.close(fig)

# ---------- 图 2：λ₁(α) 扫描 ----------
scan = np.load("figures/lambda_scan.npz")
alphas, lams = scan["alphas"], scan["lams"]
a_star = float(scan["alpha_star"])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(alphas, lams, "o-", ms=4, lw=1.4)
ax.axhline(1.0, color="r", ls="--", lw=1.4, label="λ₁ = 1（自洽线）")
ax.axvline(a_star, color="g", ls=":", lw=1.4, label=f"α* ≈ {a_star:.2f}")
ax.set_xlabel("ALPHA（BETA=1.0 固定）")
ax.set_ylabel("基模广义特征值 λ₁")
ax.set_title(r"广义特征值扫描：只有 $\lambda_1$=1 时方程才有解")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("doc_figures/lambda_scan.png", dpi=150)
plt.close(fig)

# ---------- 图 3：PINN(α=2) vs FD 参考解 ----------
pin = np.load("figures/psi_grid_consistent.npz")
ref = np.load("figures/psi_ref.npz")
Rg, Zg = pin["Rg"], pin["Zg"]
psi_p = np.nan_to_num(pin["psi"])
psi_fd = bilinear(ref["psi_f"], ref["Rf"], ref["Zf"], Rg, Zg)
psi_fd = np.where((Rg - 2.0) ** 2 + Zg ** 2 < 0.8 ** 2, psi_fd, np.nan)
diff = np.abs(psi_p - np.nan_to_num(psi_fd))
mask = (Rg - 2.0) ** 2 + Zg ** 2 < 0.8 ** 2
rel_l2 = np.sqrt(np.nansum((psi_p - psi_fd) ** 2) / np.nansum(psi_fd ** 2))
print(f"PINN(α=2) vs FD: 相对 L2 误差 = {rel_l2:.4e}")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
th = np.linspace(0, 2 * np.pi, 200)
for ax in axes:
    ax.plot(2.0 + 0.8 * np.cos(th), 0.8 * np.sin(th), "r--", lw=1.4)
    ax.plot([2.0], [0.0], "r*", ms=12)
    ax.axis("scaled")
cf0 = axes[0].contourf(Rg, Zg, psi_fd, levels=24, cmap="viridis")
axes[0].contour(Rg, Zg, psi_fd, levels=24, colors="w", linewidths=0.3)
axes[0].set_title("FD 参考解（广义特征值法）")
fig.colorbar(cf0, ax=axes[0], shrink=0.85)
cf1 = axes[1].contourf(Rg, Zg, np.where(mask, psi_p, np.nan), levels=24, cmap="viridis")
axes[1].contour(Rg, Zg, np.where(mask, psi_p, np.nan), levels=24, colors="w", linewidths=0.3)
axes[1].set_title("PINN（α=2.0, β=1.0）")
fig.colorbar(cf1, ax=axes[1], shrink=0.85)
cf2 = axes[2].contourf(Rg, Zg, np.where(mask, diff, np.nan), levels=24, cmap="magma")
axes[2].set_title(f"|PINN − FD|（相对 L2 = {rel_l2:.2e}）")
fig.colorbar(cf2, ax=axes[2], shrink=0.85)
fig.tight_layout()
fig.savefig("doc_figures/pinn_vs_ref.png", dpi=150)
plt.close(fig)

# ---------- 图 4：q(ψ) 对比 ----------
qp = np.load("figures/q_profile_consistent.npz")
qr = ref["q"]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(qp["levels"], qp["qs"], "o-", color="darkorange", lw=1.6, label="PINN（α=2, β=1）")
ax.plot(ref["levels"], qr, "s--", color="navy", lw=1.4, label="FD 参考解")
ax.set_xlabel(r"$\psi$")
ax.set_ylabel(r"$q(\psi)$")
ax.set_title("安全因子 q(ψ)：PINN 极向线积分 vs FD 参考值")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("doc_figures/q_compare.png", dpi=150)
plt.close(fig)

# ---------- 图 5：残差对比（无解 vs 自洽） ----------
r1 = np.load("figures/residual_ab11.npz")["res"]
r2 = np.load("figures/residual_consistent.npz")["res"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, res, title in [
        (axes[0], r1, f"α=β=1（无精确解）  mean|res|={np.abs(r1).mean():.2e}"),
        (axes[1], r2, f"α=2.0, β=1.0（自洽）  mean|res|={np.abs(r2).mean():.2e}")]:
    sc = ax.scatter(np.load("figures/residual_ab11.npz")["R"],
                    np.load("figures/residual_ab11.npz")["Z"],
                    c=np.abs(res), s=8, cmap="plasma",
                    vmax=np.percentile(np.abs(res), 95))
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.axis("scaled")
    fig.colorbar(sc, ax=ax, shrink=0.85)
fig.suptitle("PDE 残差空间分布：参数自洽后残差显著下降、峰值消解", y=1.02)
fig.tight_layout()
fig.savefig("doc_figures/residual_compare.png", dpi=150)
plt.close(fig)

print("插图已生成到 doc_figures/：loss_curves, lambda_scan, pinn_vs_ref, q_compare, residual_compare")
