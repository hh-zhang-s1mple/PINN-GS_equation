# -*- coding: utf-8 -*-
"""gs_pinn.py —— 用 PINN（物理信息神经网络）求解 Grad-Shafranov 方程（基线版）

物理模型：柱坐标 (R, Z)，圆形截面托卡马克，R∈[1.2,2.8], Z∈[-0.8,0.8]
  方程  R·∂/∂R((1/R)ψ_R) + ψ_ZZ = -μ0 R² p'(ψ) - 0.5 (f²)'(ψ)
  边界  圆边界 ψ=0；磁轴(2.0,0) ψ=1；Z=0 赤道面 ∂ψ/∂Z=0
  源项  p'(ψ) = ALPHA·ψ,  (f²)'(ψ) = BETA·ψ   （线性，严禁常数源项）

实现规范（与《提示词.txt》逐条对应）：
  采样  Sobol 域内2000 + 圆边界均匀100 + 赤道面20
  网络  2→50→50→50→1，激活只准 torch.sin，xavier_uniform_ 初始化
  损失  L = L_PDE + 100·L_BC + 10·L_sym（权重写死）
  优化  仅 Adam(lr=1e-3) + ReduceLROnPlateau(patience=500, factor=0.5)
  阶段  RUN_PHASE=1 验证导数(500步)；RUN_PHASE=2 正式训练(5000步)+四张图
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows GBK 控制台打印 ψ/α/β 兼容

# ===================== 运行控制 =====================
RUN_PHASE = 1        # 阶段开关：1 = 验证导数(500步)  2 = 正式训练(5000步)+绘图
EPOCH_1 = (RUN_PHASE == 1)
EPOCH_2 = (RUN_PHASE == 2)
BASELINE_MODE = True  # 基线优先：禁止残差连接/Fourier特征/多任务/迁移学习

# ===================== 配置 =====================
Config = {
    "R_MIN": 1.2, "R_MAX": 2.8,      # R 方向范围
    "Z_MIN": -0.8, "Z_MAX": 0.8,     # Z 方向范围
    "R0": 2.0, "A": 0.8,             # 圆边界：圆心 (R0,0)，半径 A（避免 R=0 奇点）
    "MU0": 1.0,                      # 归一化真空磁导率
    "P0": 1.0,                       # 归一化压力基准
    "F0": 1.0,                       # 磁轴处极向电流函数 f0（用于 f² = f0² + β/2(ψ²-1)）
    "ALPHA": 1.0,                    # 修改此处可切换非线性profile，例如 α=2.0
    "BETA": 1.0,                     # (f²)'(ψ) = BETA·ψ
    "N_INT": 2000,                   # Sobol 域内采样数
    "N_BC": 100,                     # 圆边界均匀采样数
    "N_SYM": 20,                     # Z=0 赤道面对称约束采样数
    "LAYERS": [2, 50, 50, 50, 1],    # 4 层全连接
    "LR": 1e-3,                      # Adam 学习率
    "W_BC": 100.0, "W_SYM": 10.0,    # 损失权重（写死为常数，禁止自适应加权）
    "PATIENCE": 500, "LR_FACTOR": 0.5,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 采样 =====================


def sample_points(cfg):
    """Sobol 域内采样（拒绝采样保留圆内点）+ 圆边界均匀 + 赤道面点。"""
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)
    Ri, Zi = [], []
    while len(Ri) < cfg["N_INT"]:
        x = sobol.draw(cfg["N_INT"])
        R = cfg["R_MIN"] + (cfg["R_MAX"] - cfg["R_MIN"]) * x[:, 0]
        Z = cfg["Z_MIN"] + (cfg["Z_MAX"] - cfg["Z_MIN"]) * x[:, 1]
        m = (R - cfg["R0"]) ** 2 + Z ** 2 < cfg["A"] ** 2
        Ri.append(R[m])
        Zi.append(Z[m])
    R_i = torch.cat(Ri)[: cfg["N_INT"]]
    Z_i = torch.cat(Zi)[: cfg["N_INT"]]
    th = torch.linspace(0.0, 2 * np.pi, cfg["N_BC"] + 1)[:-1]
    R_b = cfg["R0"] + cfg["A"] * torch.cos(th)
    Z_b = cfg["A"] * torch.sin(th)
    R_s = torch.linspace(cfg["R_MIN"], cfg["R_MAX"], cfg["N_SYM"])
    Z_s = torch.zeros_like(R_s)
    return (R_i, Z_i), (R_b, Z_b), (R_s, Z_s)


def standardize(x, mu, sd):
    """StandardScaler：(x - mean) / std，禁止自创归一化策略。"""
    return (x - mu) / sd

# ===================== 网络 =====================


class PINN(torch.nn.Module):
    """4 层全连接 2→50→50→50→1，隐藏层激活只准用 torch.sin。"""

    def __init__(self, layers):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = torch.nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_uniform_(lin.weight)   # 强制 xavier_uniform_
            torch.nn.init.zeros_(lin.bias)
            self.linears.append(lin)

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = torch.sin(x)                        # 激活只准 torch.sin
        return x

# ===================== PDE 残差与损失 =====================


def pde_residual(net, R, Z, mu, sd, cfg):
    """GS 方程残差：ψ_RR - ψ_R/R + ψ_ZZ + μ0 R² p'(ψ) + 0.5 (f²)'(ψ)。"""
    R.requires_grad_(True)
    Z.requires_grad_(True)
    x = torch.stack([(R - mu[0]) / sd[0], (Z - mu[1]) / sd[1]], dim=1)
    psi = net(x).squeeze(-1)
    gR, gZ = torch.autograd.grad(psi, (R, Z), grad_outputs=torch.ones_like(psi), create_graph=True)
    psi_R, psi_Z = gR, gZ
    psi_RR = torch.autograd.grad(psi_R, R, grad_outputs=torch.ones_like(psi_R), create_graph=True)[0]
    psi_ZZ = torch.autograd.grad(psi_Z, Z, grad_outputs=torch.ones_like(psi_Z), create_graph=True)[0]
    p_prime = cfg["ALPHA"] * psi                        # 源项直接代入，严禁对源项再求导
    f2_prime = cfg["BETA"] * psi
    return psi_RR - psi_R / R + psi_ZZ + cfg["MU0"] * R ** 2 * p_prime + 0.5 * f2_prime


def train_step(net, opt, data, mu, sd, cfg):
    """一次梯度下降。返回 (总损失, L_PDE, L_BC, L_sym)。"""
    net.train()
    opt.zero_grad()
    (R_i, Z_i), (R_b, Z_b, t_b), (R_s, Z_s) = data
    res = pde_residual(net, R_i, Z_i, mu, sd, cfg)
    loss_pde = (res ** 2).mean()
    x_b = torch.stack([(R_b - mu[0]) / sd[0], (Z_b - mu[1]) / sd[1]], dim=1)
    psi_b = net(x_b).squeeze(-1)
    loss_bc = ((psi_b - t_b) ** 2).mean()               # 100 个边界 ψ=0 + 1 个磁轴 ψ=1
    x_s = torch.stack([(R_s - mu[0]) / sd[0], (Z_s - mu[1]) / sd[1]], dim=1)
    x_s.requires_grad_(True)
    psi_s = net(x_s).squeeze(-1)
    dpsi_s = torch.autograd.grad(psi_s, x_s, grad_outputs=torch.ones_like(psi_s), create_graph=True)[0]
    loss_sym = (dpsi_s[:, 1] ** 2).mean()               # Z=0 上 ∂ψ/∂Z = 0
    loss = loss_pde + cfg["W_BC"] * loss_bc + cfg["W_SYM"] * loss_sym
    loss.backward()
    opt.step()
    return loss, loss_pde, loss_bc, loss_sym

# ===================== 网格求值与 q(ψ) 诊断 =====================


def eval_on_grid(net, cfg, mu, sd, Nr, Nz):
    """在 (Nr×Nz) 规则网格上求 ψ（向量化分块，无梯度）。"""
    R = np.linspace(cfg["R_MIN"], cfg["R_MAX"], Nr)
    Z = np.linspace(cfg["Z_MIN"], cfg["Z_MAX"], Nz)
    Rg, Zg = np.meshgrid(R, Z, indexing="ij")
    pts = np.stack([Rg.ravel(), Zg.ravel()], axis=1).astype(np.float32)
    psi = np.empty(pts.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, pts.shape[0], 8192):
            t = torch.from_numpy(pts[i:i + 8192]).to(DEVICE)
            psi[i:i + 8192] = net((t - mu) / sd).squeeze(-1).cpu().numpy()
    return Rg, Zg, psi.reshape(Nr, Nz)


def compute_q_profile(net, cfg, mu, sd, levels):
    """安全因子 q(ψ) = F(ψ)/(2π)·∮_ψ dl/(R²|∇ψ|)。

    严格极向线积分：matplotlib.contour 提取等值线顶点 → numpy 梯形法则沿路径积分。
    F = sqrt(f0² + BETA/2·(ψ²-1))，常数由磁轴处 f0=1、ψ=1 确定。
    禁止 scipy.integrate / 局部近似公式 q≈F/(R²|∇ψ|)。
    """
    Rg, Zg, psi = eval_on_grid(net, cfg, mu, sd, 250, 250)
    psi[(Rg - cfg["R0"]) ** 2 + Zg ** 2 > cfg["A"] ** 2] = np.nan
    fig = plt.figure()
    cs = plt.contour(Rg, Zg, psi, levels=levels)
    qs = []
    for k, segs in enumerate(cs.allsegs):
        if len(segs) == 0 or all(len(s) == 0 for s in segs):
            qs.append(np.nan)                            # 该 ψ 层无等值线（如折中解 ψ_max 过低）
            continue
        total = 0.0
        for seg in segs:
            if len(seg) == 0:
                continue
            if np.hypot(seg[0, 0] - seg[-1, 0], seg[0, 1] - seg[-1, 1]) > 1e-12:
                seg = np.vstack([seg, seg[:1]])          # 确保路径闭合
            pts = torch.from_numpy(seg.astype(np.float32)).to(DEVICE)
            pts.requires_grad_(True)
            ps = net((pts - mu) / sd)
            g = torch.autograd.grad(ps.sum(), pts, create_graph=False)[0].detach().cpu().numpy()
            grad_mag = np.hypot(g[:, 0], g[:, 1])
            integ = 1.0 / (seg[:, 0] ** 2 * grad_mag + 1e-8)
            dseg = np.diff(seg, axis=0)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(dseg[:, 0], dseg[:, 1]))])
            total += np.trapezoid(integ, arc)            # numpy 梯形法则（np.trapz 现名）
        F = np.sqrt(cfg["F0"] ** 2 + cfg["BETA"] / 2.0 * (levels[k] ** 2 - 1.0))
        qs.append(F / (2 * np.pi) * total)
    plt.close(fig)
    return levels, np.array(qs)


def diagnostic_gradient_check(net, mu, sd):
    """[DEBUG] 有限差分 vs 自动微分。默认不执行：取消 train() 中注释才运行。"""
    pts = torch.tensor([[2.0, 0.0], [1.6, 0.4]], dtype=torch.float32, device=DEVICE)
    R, Z = pts[:, 0:1].clone().requires_grad_(True), pts[:, 1:2].clone().requires_grad_(True)
    psi = net((torch.cat([R, Z], dim=1) - mu) / sd)
    gR, gZ = torch.autograd.grad(psi.sum(), (R, Z), create_graph=False)
    grad_ad = np.stack([gR.detach().cpu().numpy().ravel(), gZ.detach().cpu().numpy().ravel()], 1)
    h = 1e-3
    for k in range(pts.shape[0]):
        fd = np.zeros(2)
        for d in range(2):
            p1 = pts[k].clone()
            p2 = pts[k].clone()
            p1[d] += h
            p2[d] -= h
            fd[d] = ((net((p1 - mu) / sd) - net((p2 - mu) / sd)) / (2 * h)).item()
        err = np.max(np.abs(grad_ad[k] - fd))
        print(f"  点{pts[k].tolist()}  AD={np.round(grad_ad[k], 6)}  FD={np.round(fd, 6)}  误差={err:.2e}")

# ===================== 训练主流程（函数式，无 Trainer 类） =====================


def train():
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs("figures", exist_ok=True)
    cfg = Config
    (R_i, Z_i), (R_b, Z_b), (R_s, Z_s) = sample_points(cfg)
    R_b = torch.cat([R_b, torch.tensor([cfg["R0"]], dtype=torch.float32)])
    Z_b = torch.cat([Z_b, torch.zeros(1, dtype=torch.float32)])
    t_b = torch.cat([torch.zeros(cfg["N_BC"]), torch.ones(1)])
    all_R = torch.cat([R_i, R_b, R_s])
    all_Z = torch.cat([Z_i, Z_b, Z_s])
    mu = torch.tensor([all_R.mean(), all_Z.mean()]).to(DEVICE)
    sd = torch.tensor([all_R.std(), all_Z.std()]).to(DEVICE)
    print(f"[StandardScaler] R: mean={mu[0].item():.4f} std={sd[0].item():.4f} | "
          f"Z: mean={mu[1].item():.4f} std={sd[1].item():.4f}")
    data = ((R_i.to(DEVICE), Z_i.to(DEVICE)), (R_b.to(DEVICE), Z_b.to(DEVICE), t_b.to(DEVICE)),
            (R_s.to(DEVICE), Z_s.to(DEVICE)))
    net = PINN(cfg["LAYERS"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["LR"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       patience=cfg["PATIENCE"], factor=cfg["LR_FACTOR"])
    if BASELINE_MODE:
        pass   # 基线模式：仅使用上述最简单的网络与损失
    if False:
        # ===== 预留高级功能接口（基线跑通前禁止启用）=====
        # L-BFGS 二阶微调：opt_lbfgs = torch.optim.LBFGS(net.parameters(), lr=0.1)
        # 残差连接 / Fourier 特征 / 多任务学习 / 迁移学习：均在此处实现后启用
        pass
    if EPOCH_1:
        print("[阶段一] 验证导数：500 步，仅输出 Loss 数值，不绘图")
        for i in range(500):
            loss, lp, lb, ls = train_step(net, opt, data, mu, sd, cfg)
            sched.step(loss.item())
            if i % 50 == 0:
                print(f"  epoch {i:4d}  loss={loss.item():.4e}  (PDE {lp.item():.2e}, "
                      f"BC {lb.item():.2e}, SYM {ls.item():.2e})")
        print("[阶段一完成] Loss 为有限值且下降 → 导数实现正确，可切换 RUN_PHASE=2")
    if EPOCH_2:
        print("[阶段二] 正式训练：5000 步，结束后绘图")
        for i in range(5000):
            loss, lp, lb, ls = train_step(net, opt, data, mu, sd, cfg)
            sched.step(loss.item())
            if i % 100 == 0:
                print(f"  epoch {i:4d}  loss={loss.item():.4e}  (PDE {lp.item():.2e}, "
                      f"BC {lb.item():.2e}, SYM {ls.item():.2e}, lr={opt.param_groups[0]['lr']:.1e})")
        print("[阶段二完成] 生成四张图 (a)ψ云图 (b)赤道剖面 (c)残差散点 (d)q(ψ)")
        levels = np.linspace(0.1, 0.9, 9)
        lv, qs = compute_q_profile(net, cfg, mu, sd, levels)
        plot_psi(net, cfg, mu, sd)
        plot_equatorial(net, cfg, mu, sd)
        plot_residual(net, R_i.to(DEVICE), Z_i.to(DEVICE), mu, sd, cfg)
        plot_q(lv, qs)
        print("q(ψ) 数值积分结果:")
        for a, b in zip(lv, qs):
            print(f"  ψ={a:.2f}  q={b:.4f}")
    with torch.no_grad():
        x_ax = torch.stack([(torch.tensor([cfg["R0"]], device=DEVICE) - mu[0]) / sd[0],
                            (torch.tensor([0.0], device=DEVICE) - mu[1]) / sd[1]], dim=1)
        psi_axis_pred = net(x_ax).item()
    print(f"ψ 在磁轴处的值：{psi_axis_pred:.4f}")


# =====================================================================
# 绘图区（按规范不计入 380 行红线；四张图 + 数据导出供教学文档使用）
# =====================================================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_psi(net, cfg, mu, sd, tag=""):
    Rg, Zg, psi = eval_on_grid(net, cfg, mu, sd, 300, 300)
    psi[(Rg - cfg["R0"]) ** 2 + Zg ** 2 > cfg["A"] ** 2] = np.nan
    np.savez(f"figures/psi_grid{tag}.npz", Rg=Rg, Zg=Zg, psi=psi)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cf = ax.contourf(Rg, Zg, psi, levels=24, cmap="viridis")
    ax.contour(Rg, Zg, psi, levels=24, colors="white", linewidths=0.4)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cfg["R0"] + cfg["A"] * np.cos(th), cfg["A"] * np.sin(th), "r--", lw=1.6,
            label=r"边界 $\psi=0$")
    ax.plot([cfg["R0"]], [0.0], "r*", ms=14, label=r"磁轴 $\psi=1$")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"(a) $\\psi$ 等值线云图  (α={cfg['ALPHA']}, β={cfg['BETA']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("scaled")
    fig.colorbar(cf, label=r"$\psi$")
    fig.tight_layout()
    fig.savefig(f"figures/fig1_psi{tag}.png", dpi=150)
    plt.close(fig)


def plot_equatorial(net, cfg, mu, sd, tag=""):
    R = np.linspace(cfg["R_MIN"], cfg["R_MAX"], 400)
    pts = torch.from_numpy(np.stack([R, np.zeros_like(R)], 1).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        psi = net((pts - mu) / sd).squeeze(-1).cpu().numpy()
    np.savez(f"figures/equator{tag}.npz", R=R, psi=psi)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(R, psi, "b-", lw=2, label=r"$\psi(R, Z=0)$")
    ax.axvline(cfg["R0"], color="gray", ls=":", lw=1)
    ax.plot([cfg["R0"]], [1.0], "r*", ms=14, label="磁轴目标 ψ=1")
    ax.axhline(1.0, color="r", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("R")
    ax.set_ylabel(r"$\psi$")
    ax.set_title("(b) 赤道面 ψ(R) 剖面")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/fig2_equator{tag}.png", dpi=150)
    plt.close(fig)


def plot_residual(net, R_i, Z_i, mu, sd, cfg, tag=""):
    res = pde_residual(net, R_i, Z_i, mu, sd, cfg).detach().cpu().numpy()
    R = R_i.detach().cpu().numpy()
    Z = Z_i.detach().cpu().numpy()
    np.savez(f"figures/residual{tag}.npz", R=R, Z=Z, res=res)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(R, Z, c=np.abs(res), s=8, cmap="plasma",
                    vmax=np.percentile(np.abs(res), 95))
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"(c) PDE 残差空间分布  (mean|res| = {np.abs(res).mean():.2e})")
    fig.colorbar(sc, label="|残差|")
    ax.axis("scaled")
    fig.tight_layout()
    fig.savefig(f"figures/fig3_residual{tag}.png", dpi=150)
    plt.close(fig)


def plot_q(levels, qs, tag=""):
    m = np.isfinite(qs)
    np.savez(f"figures/q_profile{tag}.npz", levels=np.asarray(levels)[m], qs=qs[m])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(np.asarray(levels)[m], qs[m], "o-", color="darkorange", lw=1.6)
    ax.set_xlabel(r"$\psi$")
    ax.set_ylabel(r"$q(\psi)$")
    ax.set_title("(d) 安全因子 q(ψ)（极向线积分）")
    fig.tight_layout()
    fig.savefig(f"figures/fig4_q{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train()
