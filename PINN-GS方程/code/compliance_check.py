# -*- coding: utf-8 -*-
"""compliance_check.py —— 对照《提示词.txt》自动检查 gs_pinn.py 是否合规

用法：python compliance_check.py
退出码 0 = 全部通过；1 = 存在未通过项（打印 FAIL 明细）。
"""
import sys

SRC = open("gs_pinn.py", encoding="utf-8").read()
lines = SRC.splitlines()

# ---- 行数红线：主体代码不含空行/注释/绘图区，严格 ≤380 ----
in_plot = False
n_spec = n_strict = 0
for ln in lines:
    s = ln.strip()
    if "绘图区（按规范不计入 380 行红线）" in s:
        in_plot = True
    if in_plot:
        continue
    if s and not s.startswith("#"):
        n_spec += 1
    if s:
        n_strict += 1
main_src = SRC.split("绘图区（按规范不计入 380 行红线）")[0]
train_src = main_src[main_src.index("def train("):]

checks = {
    "行数规范口径(不含空行/注释/绘图)<=380": n_spec <= 380,
    "行数严格口径(含注释)<=380": n_strict <= 380,
    "SobolEngine 采样": "torch.quasirandom.SobolEngine" in SRC,
    "xavier_uniform_ 初始化": "xavier_uniform_" in SRC,
    "激活只准 torch.sin": "torch.sin" in SRC and "relu" not in SRC.lower() and "swish" not in SRC.lower(),
    "优化器仅 Adam": "torch.optim.Adam" in SRC,
    "ReduceLROnPlateau(patience=500, factor=0.5)": ("ReduceLROnPlateau" in SRC and "500" in SRC and "0.5" in SRC),
    "损失权重 100/10 写死": all(k in SRC for k in ["W_BC", "100.0", "10.0"]),
    "BASELINE_MODE 开关": "BASELINE_MODE" in SRC,
    "高级功能锁在 if False": "if False:" in main_src,
    "两阶段 RUN_PHASE/EPOCH_1/EPOCH_2": all(k in SRC for k in ["RUN_PHASE", "EPOCH_1", "EPOCH_2"]),
    "500/5000 步两阶段循环": "range(500)" in SRC and "range(5000)" in SRC,
    "禁止 scipy（无 import/from 导入）": "import scipy" not in SRC and "from scipy" not in SRC,
    "禁止 skimage": "skimage" not in SRC,
    "train() 内无 try-except": "try" not in train_src.split("def plot_psi")[0],
    "未提及 D 形边界": "D形" not in SRC and "D 形" not in SRC,
    "q 用 numpy 梯形法则(np.trapezoid)": "np.trapezoid" in SRC,
    "q 用 matplotlib.contour 提取等值线": "plt.contour" in SRC,
    "q 函数在主代码(绘图区之前)": "def compute_q_profile" in main_src,
    "磁轴打印格式": "ψ 在磁轴处的值：{psi_axis_pred:.4f}" in SRC,
    "Config 含 ALPHA/BETA 及切换注释": '"ALPHA"' in SRC and "例如 α=2.0" in SRC,
    "StandardScaler (x-mean)/std": "standardize" in SRC and "(x - mu) / sd" in SRC,
    "源项直接代入(不对源项求导)": 'p_prime = cfg["ALPHA"] * psi' in SRC,
    "诊断函数 diagnostic_gradient_check": "diagnostic_gradient_check" in SRC,
    "采样数 2000/100/20": all(k in SRC for k in ['"N_INT": 2000', '"N_BC": 100', '"N_SYM": 20']),
    "网络结构 [2,50,50,50,1]": "[2, 50, 50, 50, 1]" in SRC,
    "四张图函数齐全": all(k in SRC for k in ["def plot_psi", "def plot_equatorial", "def plot_residual", "def plot_q"]),
}
bad = [k for k, v in checks.items() if not v]
print(f"规范口径行数: {n_spec}  | 严格口径行数(含注释): {n_strict}")
print(f"检查项: {len(checks)}  通过: {len(checks) - len(bad)}")
for b in bad:
    print("  FAIL:", b)
print("全部通过 ✓" if not bad else "存在未通过项 ✗")
sys.exit(1 if bad else 0)
