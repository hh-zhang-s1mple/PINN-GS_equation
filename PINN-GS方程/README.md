# PINN-GS方程：物理信息神经网络求解托卡马克 Grad-Shafranov 方程

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#许可证)

本项目使用**物理信息神经网络（Physics-Informed Neural Network, PINN）**求解柱坐标下的 Grad-Shafranov（GS）方程，实现圆形截面托卡马克的等离子体平衡重构。项目同时提供有限差分（FD）广义特征值参考解，用于验证 PINN 结果的正确性。

---

## 目录

- [项目背景](#项目背景)
- [物理模型](#物理模型)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [关键结果](#关键结果)
- [自洽性分析](#自洽性分析)
- [规范与合规检查](#规范与合规检查)
- [许可证](#许可证)

---

## 项目背景

Grad-Shafranov 方程是描述轴对称托卡马克等离子体平衡的核心方程。传统数值方法（有限差分、有限元）需要离散网格并求解大型线性/非线性方程组。PINN 将 PDE 残差作为损失函数的一部分，利用神经网络的自动微分能力直接在连续空间中逼近解，无需网格生成，且天然满足物理约束。

本项目实现了一个**基线版 PINN 求解器**，重点关注：

- GS 方程残差的精确自动微分实现
- 边界条件与对称性约束的嵌入
- 安全因子 \(q(\psi)\) 的极向线积分计算
- 与有限差分参考解的定量对比
- 方程参数自洽性（广义特征值条件）的分析

---

## 物理模型

### 求解域

柱坐标 \((R, Z)\)，圆形截面托卡马克：

- \(R \in [1.2,\ 2.8]\)，\(Z \in [-0.8,\ 0.8]\)
- 圆形边界：圆心 \((R_0=2.0,\ Z=0)\)，半径 \(A=0.8\)（避开 \(R=0\) 奇点）

### Grad-Shafranov 方程

\[
R\frac{\partial}{\partial R}\left(\frac{1}{R}\psi_R\right) + \psi_{ZZ} = -\mu_0 R^2 p'(\psi) - \frac{1}{2}(f^2)'(\psi)
\]

展开为：

\[
\psi_{RR} - \frac{\psi_R}{R} + \psi_{ZZ} + \mu_0 R^2 p'(\psi) + \frac{1}{2}(f^2)'(\psi) = 0
\]

### 边界与约束条件

| 条件 | 位置 | 取值 |
|------|------|------|
| Dirichlet 边界 | 圆形边界 | \(\psi = 0\) |
| 磁轴约束 | \((R_0, 0) = (2.0, 0)\) | \(\psi = 1\) |
| 对称性（Neumann） | 赤道面 \(Z=0\) | \(\partial\psi/\partial Z = 0\) |

### 源项（线性 profile）

\[
p'(\psi) = \alpha \cdot \psi, \qquad (f^2)'(\psi) = \beta \cdot \psi
\]

其中 \(\alpha\)（ALPHA）和 \(\beta\)（BETA）为可调参数。默认 \(\alpha = \beta = 1.0\)，自洽参数约为 \(\alpha^* \approx 2.13,\ \beta = 1.0\)。

---

## 项目结构

```
PINN-GS方程/
├── gs_pinn.py              # PINN 主程序（网络定义、PDE残差、训练、绘图）
├── gs_reference.py         # 有限差分参考解（广义特征值法 + q(ψ) 积分）
├── compliance_check.py     # 规范合规自动检查脚本
├── make_doc_figures.py     # 教学文档插图生成（读取日志与 npz 数据）
├── requirements.txt        # Python 依赖清单
├── training_log_phase1.txt          # 阶段一训练日志（500 步导数验证）
├── training_log_phase2.txt          # 阶段二训练日志
├── training_log_phase2_ab11.txt     # α=β=1 运行日志（无精确解 → 折中地板）
├── training_log_phase2_consistent.txt # α=2.0,β=1.0 自洽运行日志
├── figures/                # 运行时生成：训练输出图与 npz 数据
│   ├── fig1_psi.png        # (a) ψ 等值线云图
│   ├── fig2_equator.png    # (b) 赤道面 ψ(R) 剖面
│   ├── fig3_residual.png   # (c) PDE 残差空间分布
│   ├── fig4_q.png          # (d) 安全因子 q(ψ)
│   └── *.npz               # 网格数据（供对比与文档使用）
└── doc_figures/            # 运行时生成：文档对比插图
    ├── loss_curves.png
    ├── lambda_scan.png
    ├── pinn_vs_ref.png
    ├── q_compare.png
    └── residual_compare.png
```

> `figures/` 和 `doc_figures/` 目录在程序运行时自动创建。

---

## 环境依赖

### 软件要求

- Python 3.8+
- CUDA（可选，GPU 加速；无 GPU 时自动回退 CPU）

### Python 包

```bash
pip install -r requirements.txt
```

依赖清单：

| 包 | 最低版本 | 用途 |
|----|----------|------|
| `torch` | 2.0 | 神经网络、自动微分、Sobol 采样 |
| `numpy` | 1.24 | 数值计算、梯形积分、网格操作 |
| `matplotlib` | 3.6 | 等值线图、剖面图、残差散点图 |

> 项目刻意不依赖 `scipy`、`scikit-image` 等库，所有积分与插值均使用 NumPy 手动实现，以保证教学透明性。

---

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd PINN-GS方程
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 阶段一：验证导数实现（500 步）

编辑 `gs_pinn.py`，设置：

```python
RUN_PHASE = 1
```

然后运行：

```bash
python gs_pinn.py
```

阶段一仅输出 Loss 数值，不绘图。若 Loss 为有限值且持续下降，说明自动微分与 PDE 残差实现正确，可进入阶段二。

### 4. 阶段二：正式训练（5000 步 + 绘图）

编辑 `gs_pinn.py`，设置：

```python
RUN_PHASE = 2
```

然后运行：

```bash
python gs_pinn.py
```

训练结束后自动在 `figures/` 目录生成四张结果图，并输出磁轴处 \(\psi\) 值与 \(q(\psi)\) 数值积分结果。

### 5. 生成有限差分参考解

```bash
# 默认 α=β=1.0
python gs_reference.py

# 指定参数
python gs_reference.py 2.13 1.0

# 扫描 α 找 λ₁=1 的自洽值（固定 β=1.0）
python gs_reference.py scan
```

---

## 使用说明

### PINN 配置参数

所有超参数集中在 `gs_pinn.py` 的 `Config` 字典中：

```python
Config = {
    "R_MIN": 1.2, "R_MAX": 2.8,        # R 方向范围
    "Z_MIN": -0.8, "Z_MAX": 0.8,       # Z 方向范围
    "R0": 2.0, "A": 0.8,               # 圆边界圆心与半径
    "MU0": 1.0,                         # 归一化真空磁导率
    "ALPHA": 1.0,                       # p'(ψ) = ALPHA·ψ
    "BETA": 1.0,                        # (f²)'(ψ) = BETA·ψ
    "N_INT": 2000,                      # Sobol 域内采样数
    "N_BC": 100,                        # 圆边界采样数
    "N_SYM": 20,                        # 赤道面对称约束采样数
    "LAYERS": [2, 50, 50, 50, 1],      # 网络结构
    "LR": 1e-3,                         # Adam 学习率
    "W_BC": 100.0, "W_SYM": 10.0,      # 损失权重
    "PATIENCE": 500, "LR_FACTOR": 0.5, # 学习率调度
}
```

### 网络架构

- **结构**：4 层全连接，\(2 \to 50 \to 50 \to 50 \to 1\)
- **激活函数**：隐藏层统一使用 `torch.sin`（正弦激活，适合周期性/光滑解）
- **初始化**：`xavier_uniform_` 权重初始化，偏置置零
- **输入标准化**：StandardScaler，\((x - \mu) / \sigma\)

### 损失函数

\[
\mathcal{L} = \mathcal{L}_{\text{PDE}} + 100 \cdot \mathcal{L}_{\text{BC}} + 10 \cdot \mathcal{L}_{\text{sym}}
\]

- \(\mathcal{L}_{\text{PDE}}\)：域内 PDE 残差均方误差（2000 个 Sobol 点）
- \(\mathcal{L}_{\text{BC}}\)：边界 \(\psi=0\) + 磁轴 \(\psi=1\) 的均方误差（101 个点）
- \(\mathcal{L}_{\text{sym}}\)：赤道面 \(\partial\psi/\partial Z=0\) 的均方误差（20 个点）

### 优化器

- **主优化器**：Adam（lr=1e-3）
- **学习率调度**：ReduceLROnPlateau（patience=500, factor=0.5）
- **训练轮次**：阶段一 500 步，阶段二 5000 步

### 安全因子 q(ψ) 计算

安全因子通过严格极向线积分计算：

\[
q(\psi) = \frac{F(\psi)}{2\pi} \oint_\psi \frac{dl}{R^2 |\nabla\psi|}
\]

其中 \(F(\psi) = \sqrt{f_0^2 + \frac{\beta}{2}(\psi^2 - 1)}\)，\(f_0=1\)。

实现步骤：
1. 在 250×250 网格上求 \(\psi\)
2. 用 `matplotlib.contour` 提取各 \(\psi\) 等值线顶点
3. 自动微分求等值线上 \(|\nabla\psi|\)
4. NumPy 梯形法则沿闭合路径积分

### 切换自洽参数

要获得方程的精确非平凡解，需将 `ALPHA` 设为自洽值（约 2.13，固定 BETA=1.0）：

```python
"ALPHA": 2.13,   # 自洽值，λ₁ ≈ 1
"BETA": 1.0,
```

也可通过 `python gs_reference.py scan` 自动扫描得到更精确的 \(\alpha^*\)。

---

## 关键结果

### 自洽参数（α=2.0, β=1.0）阶段二训练

| 指标 | 数值 |
|------|------|
| 最终总损失 | \(\approx 1.20 \times 10^{-2}\) |
| PDE 损失 | \(\approx 7.8 \times 10^{-3}\) |
| 边界损失 | \(\approx 4.2 \times 10^{-5}\) |
| 对称损失 | \(\approx 4.6 \times 10^{-8}\) |
| 磁轴 \(\psi(2.0, 0)\) | **0.9880**（目标 1.0） |

### 安全因子 q(ψ)

| \(\psi\) | \(q(\psi)\) |
|----------|-------------|
| 0.10 | 0.1175 |
| 0.20 | 0.0913 |
| 0.30 | 0.0752 |
| 0.40 | 0.0646 |
| 0.50 | 0.0572 |
| 0.60 | 0.0519 |
| 0.70 | 0.0480 |
| 0.80 | 0.0450 |
| 0.90 | 0.0427 |

### 输出图说明

阶段二训练结束后自动生成四张图：

- **(a) ψ 等值线云图**：求解域内磁面分布，红色虚线为边界，红星为磁轴
- **(b) 赤道面 ψ(R) 剖面**：\(Z=0\) 上 \(\psi\) 随 \(R\) 的变化，验证磁峰值位置
- **(c) PDE 残差空间分布**：域内各采样点的残差绝对值散点图
- **(d) 安全因子 q(ψ)**：极向线积分得到的安全因子随磁面的变化

---

## 自洽性分析

本项目的一个核心发现是：**GS 方程在线性源项下并非对任意参数都有非平凡解**。

### 广义特征值问题

当 \(p'(\psi) = \alpha\psi\)，\((f^2)'(\psi) = \beta\psi\) 时，GS 方程化为：

\[
-\Delta^* \psi = c(R) \cdot \psi, \qquad c(R) = \mu_0 R^2 \alpha + \frac{\beta}{2}
\]

这是一个广义特征值问题 \(-\Delta^* \psi = \lambda \cdot c(R) \cdot \psi\)。仅当基模特征值 \(\lambda_1 = 1\) 时，方程才有满足"磁轴 \(\psi=1\)"的非平凡解；若 \(\lambda_1 \neq 1\)，则只有零解，PINN 只能得到**折中解**（损失下降到某个地板值后停滞）。

### 两种参数对比

| 参数 | \(\lambda_1\) | 解的存在性 | PINN 表现 |
|------|---------------|-----------|-----------|
| \(\alpha=\beta=1.0\) | \(\neq 1\) | 无精确解（仅零解） | 损失降至地板值后停滞，磁轴 \(\psi \ll 1\) |
| \(\alpha\approx2.13,\ \beta=1.0\) | \(\approx 1\) | 存在非平凡解 | 损失持续下降，磁轴 \(\psi \approx 1\) |

### 扫描自洽参数

运行以下命令可自动扫描找到 \(\lambda_1=1\) 对应的 \(\alpha^*\)：

```bash
python gs_reference.py scan
```

输出示例：

```
[扫描] 固定 BETA=1.0，λ₁=1 的自洽 ALPHA* = 2.1300 (λ₁=1.0000)
```

---

## 规范与合规检查

项目提供 `compliance_check.py` 脚本，用于自动检查 `gs_pinn.py` 是否符合实现规范。检查项包括：

- 行数限制（主体代码 ≤ 380 行，不含绘图区）
- SobolEngine 采样
- xavier_uniform_ 初始化
- 激活函数仅使用 torch.sin
- 优化器仅 Adam + ReduceLROnPlateau
- 损失权重 100/10 写死
- 两阶段训练（500/5000 步）
- 禁止 scipy / skimage 依赖
- q(ψ) 使用 numpy 梯形法则 + matplotlib.contour
- 四张图函数齐全
- 源项直接代入（不对源项再求导）
- StandardScaler 标准化

运行检查：

```bash
python compliance_check.py
```

退出码 0 表示全部通过，1 表示存在未通过项。

---

## 许可证

本项目采用 [MIT 许可证](https://opensource.org/licenses/MIT)。详见项目根目录下的 `LICENSE` 文件。

---

## 致谢

本项目受物理信息神经网络（PINN）方法启发，参考了 Raissi 等人的开创性工作。Grad-Shafranov 方程的理论框架基于标准托卡马克等离子体物理教材。
