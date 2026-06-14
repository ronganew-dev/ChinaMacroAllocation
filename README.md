# ChinaMacroAllocation 🇨🇳📊

> 宏观周期驱动的多资产配置引擎 + 通用量化算子库

---

## 项目总览

**ChinaMacroAllocation** 是一个基于中国宏观经济周期信号的量化资产配置框架。它将周期信号映射到多资产组合的风险预算，支持 EWMA 风险建模、子组合控制与动态再平衡。

项目同时提炼出一套 **通用量化算子库** (`operators/`)，可独立复用于趋势跟踪、动量分析、波动率建模等场景。

### 核心框架

```
┌─────────────────────────────────────────────────┐
│                 ChinaMacroAllocation              │
│  ┌─────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  strategy/   │  │ analysis/  │  │operators/ │ │
│  │  宏观配置策略  │  │  分析工具   │  │  通用算子   │ │
│  └──────┬──────┘  └─────┬──────┘  └─────┬─────┘ │
│         │                │               │        │
│         └────────────────┼───────────────┘        │
│                    可插拔算子架构                    │
└─────────────────────────────────────────────────┘
```

---

## 目录结构

```
ChinaMacroAllocation/
│
├── operators/                    # 🔧 通用量化算子库（可独立复用）
│   ├── __init__.py
│   ├── base.py                   # nanmask 装饰器 + 类型别名
│   ├── momentum.py               # 动量/趋势 (ACF, SNR, Hurst, 均线回测)
│   ├── volatility.py             # 波动率/风险 (EWMA协方差, 回撤, Ulcer Index)
│   ├── cross_sectional.py        # 横截面 (排名、标准化、复合评分、行业中性化)
│   └── volume.py                 # 🔄 量价结合 (开发中)
│
├── strategy/                     # 📈 宏观周期配置策略
│   ├── __init__.py
│   ├── config.py                 # 全局参数 (资产、风险、调仓)
│   └── utils.py                  # 工具函数
│
├── analysis/                     # 📊 资产分析模块
│   ├── __init__.py
│   └── trend_follow.py           # 趋势跟踪适合度分析
│
├── tests/                        # ✅ 单元测试
│   ├── __init__.py
│   ├── test_momentum.py
│   └── test_volatility.py
│
├── data/                         # 📁 数据目录 (gitignored)
├── requirements.txt              # 📦 极简依赖
└── README.md
```

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行趋势分析

```bash
python -m analysis.trend_follow
```

### 运行测试

```bash
pytest tests/ -v
```

---

## 🧮 算子手册 (Formulaic Operators)

所有算子遵循 **WorldQuant BRAIN Formulaic Operators** 工程规范：
- **纯函数式** — 无类、无状态、无 `fit/transform`
- **全程向量化** — 零显式 Python 数据元素循环
- **原子粒度** — 每个函数 = 一个公式

### 动量/趋势类 (`operators/momentum.py`)

| 函数 | 说明 | 公式 |
|------|------|------|
| `acf()` | 单阶滞后自相关 | $\rho(k) = \text{Corr}(r_t, r_{t-k})$ |
| `acf_multi()` | 多阶自相关聚合 | — |
| `snr()` | 信噪比 | $\text{SNR} = \mu_{\text{ann}} / \sigma_{\text{ann}}$ |
| `hurst_exponent()` | Hurst 指数 | R/S 重标极差分析（纯向量化） |
| `streaks()` | 连涨连跌极值 | 最长连续正/负收益天数 |
| `trend_efficiency()` | 趋势效率 | $\eta = [\prod(1+r)-1] / \sum\|r\|$ |
| `trend_strength()` | 趋势强度 | $\| \prod(1+r)-1 \|$ |
| `trend_stability()` | 趋势稳定性 | 快慢均线同向比例 |
| `ma_crossover_metrics()` | 均线交叉策略绩效 | 20/60 SMA crossover |

### 波动率/风险类 (`operators/volatility.py`)

| 函数 | 说明 | 公式 |
|------|------|------|
| `ewma_weights()` | EWMA 衰减权重 | $w_i = (1-\lambda)\lambda^{n-1-i}$ |
| `ewma_cov()` | 指数加权协方差矩阵 | $\Sigma = \sum w_i (x_i-\mu)^T(x_i-\mu)$ |
| `drawdown_series()` | 回撤时间序列 | $DD(t)=NAV(t)/\max_{s\leq t}NAV(s)-1$ |
| `max_drawdown()` | 最大回撤 | $\min_t DD(t)$ |
| `drawdown_details()` | 回撤峰值 / 谷底 / 恢复详情 | — |
| `recovery_time()` | 恢复交易日数 | — |
| `ulcer_index()` | Ulcer 指数 | $\sqrt{\frac{1}{N}\sum DD(t)^2}$ |

### 横截面类 (`operators/cross_sectional.py`)

| 函数 | 说明 |
|------|------|
| `cross_sectional_rank()` | 多资产多指标排名（方向可配） |
| `cross_sectional_zscore()` | 横截面 Z-Score 标准化 |
| `composite_score()` | 加权复合评分（矩阵向量化） |
| `sector_neutralize()` | 行业中性化（去均值 / 标准化） |

### 快速使用

```python
from operators import snr, hurst_exponent, drawdown_details, composite_score

# 时序算子 — 传入 Series，返回数值
signal = snr(return_series)
h = hurst_exponent(return_series)

# 回撤分析 — 传入净值 Series
dd = drawdown_details((1 + return_series).cumprod())
print(dd["max_drawdown"], dd["recovery_date"])

# 横截面排名 — 传入 DataFrame，行=资产、列=指标
ranks = cross_sectional_rank(metrics_df, {"SNR": True, "Hurst": True})
scored = composite_score(ranks, {"SNR": 0.5, "Hurst": 0.5})
```

---

## 🏗️ 构建哲学

### Formulaic Operators

本项目的算子库严格遵循 **WorldQuant BRAIN Formulaic Operators** 工程规范：

1. **纯函数式** — 没有类、没有 `fit/transform`、没有内部状态。每个算子就是一个纯函数：`f(Series) → value`
2. **全程向量化** — 所有算子通过 pandas/numpy 的向量化操作实现，零显式 Python `for` 循环操作数据元素
3. **原子粒度** — 每个函数只完成一个公式的计算（例如 `snr()` 只算信噪比，不包含排名逻辑）
4. **pandas-first** — 所有时序算子直接接受 `pd.Series` 作为输入，保持索引对齐

这种设计的直接好处：**可单元测试、可组合、可复用**。

### 渐进式重构

项目采用"先建立结构，再填充内容"的策略。第一版建立完整的算子框架和测试体系，后续按模块逐步：

1. ✅ 算子库骨架 + 现有代码提炼
2. ✅ Formulaic Operators 工程化重写
3. 🔄 `strategy/` 回测引擎完整化
4. ⏳ `volume.py` 量价算子实现

---

## 技术栈

| 组件 | 选型 |
|------|------|
| 数据处理 | pandas, numpy |
| 统计分析 | scipy |
| Excel I/O | openpyxl |
| 测试 | pytest |

---

## 许可

MIT License
