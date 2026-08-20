# PINN-GS方程：物理信息神经网络求解托卡马克 Grad-Shafranov 方程

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green)](#许可证)

使用物理信息神经网络（PINN）求解柱坐标下的 Grad-Shafranov 方程，实现圆形截面托卡马克等离子体平衡重构，同时提供有限差分（FD）广义特征值参考解用于验证。

## 物理模型

- **求解域**：圆形截面，圆心 (R₀=2.0, Z=0)，半径 A=0.8，R∈[1.2,2.8]，Z∈[-0.8,0.8]
- **GS 方程**：ψ_RR − ψ_R/R + ψ_ZZ + μ₀R²p'(ψ) + 0.5(f²)'(ψ) = 0
- **边界条件**：圆边界 ψ=0，磁轴 (2.0,0) ψ=1，赤道面 ∂ψ/∂Z=0
- **源项**：p'(ψ)=α·ψ，(f²)'(ψ)=β·ψ（线性 profile）

> 线性源项下 GS 方程化为广义特征值问题，仅当基模 λ₁=1 时存在非平凡解。α=β=1 时无精确解，自洽参数约为 α*≈2.13（β=1）。

## 项目结构

```
PINN-GS方程/
├── gs_pinn.py           # PINN 主程序（网络、PDE残差、训练、绘图）
├── gs_reference.py      # FD 参考解（广义特征值法 + q(ψ) 积分）
├── requirements.txt     # 依赖清单
└── training_log_*.txt   # 训练日志
```

运行时自动生成 `figures/`（结果图与 npz 数据）和 `doc_figures/`。

## 环境依赖

```bash
pip install -r requirements.txt
```

依赖 `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.6`，不依赖 scipy（积分与插值均用 NumPy 实现）。

## 快速开始

编辑 `gs_pinn.py` 切换 `RUN_PHASE`，然后运行：

```bash
python gs_pinn.py
```

- **RUN_PHASE=1**：500 步导数验证，仅输出 Loss，不绘图
- **RUN_PHASE=2**：5000 步正式训练，结束后生成四张结果图

阶段二输出：(a) ψ 等值线云图、(b) 赤道面 ψ(R) 剖面、(c) PDE 残差分布、(d) 安全因子 q(ψ)。

生成 FD 参考解：

```bash
python gs_reference.py            # 默认 α=β=1
python gs_reference.py 2.13 1.0  # 指定参数
python gs_reference.py scan       # 扫描 α 找 λ₁=1 的自洽值
```

## PINN 配置

- **网络**：2→50→50→50→1 全连接，隐藏层激活 `torch.sin`，xavier_uniform_ 初始化
- **采样**：Sobol 域内 2000 点 + 圆边界 100 点 + 赤道面 20 点
- **损失**：L = L_PDE + 100·L_BC + 10·L_sym
- **优化器**：Adam(lr=1e-3) + ReduceLROnPlateau(patience=500, factor=0.5)
- **q(ψ)**：contour 提取等值线 + NumPy 梯形法则极向线积分

所有超参数集中在 `Config` 字典中，切换自洽参数修改 `"ALPHA": 2.13` 即可。

## 关键结果（α=2.0, β=1.0）

| 指标 | 数值 |
|------|------|
| 最终总损失 | ≈ 1.20×10⁻² |
| 磁轴 ψ(2.0, 0) | 0.9880（目标 1.0） |
| 边界损失 | ≈ 4.2×10⁻⁵ |



## 许可证

[MIT License](https://opensource.org/licenses/MIT)

