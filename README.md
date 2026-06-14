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
│   ├── base.py                   # TimeSeriesOperator 抽象基类
│   ├── momentum.py               # 动量/趋势 (ACF, SNR, Hurst, 均线回测)
│   ├── volatility.py             # 波动率/风险 (EWMA协方差, 回撤)
│   ├── cross_sectional.py        # 横截面 (排名、复合评分)
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

## 🧮 算子手册 (Operators)

### 动量/趋势类 (`operators/momentum.py`)

| 函数 | 说明 | 数学意义 |
|------|------|----------|
| `compute_autocorr()` | 滞后自相关 | $\rho(k) = \text{Corr}(r_t, r_{t-k})$ |
| `compute_snr()` | 信噪比 | $\text{SNR} = \frac{\mu_{\text{ann}}}{\sigma_{\text{ann}}}$ |
| `compute_hurst()` | Hurst 指数 | R/S 重标极差分析 |
| `compute_streaks()` | 连涨连跌极值 | 最长连续正/负收益天数 |
| `compute_trend_efficiency()` | 趋势效率 | $\eta = \frac{\prod(1+r)-1}{\sum|r|}$ |
| `compute_trend_stability()` | 趋势稳定性 | 快慢均线同向天数比例 |
| `compute_ma_backtest()` | 双均线策略回测 | 20/60 SMA 交叉 |

### 波动率/风险类 (`operators/volatility.py`)

| 函数 | 说明 |
|------|------|
| `calc_ewma_cov()` | 指数加权协方差矩阵 |
| `calc_drawdowns()` | 最大回撤 + 峰值/谷底/恢复日期 |

### 横截面类 (`operators/cross_sectional.py`)

| 函数 | 说明 |
|------|------|
| `cross_sectional_rank()` | 多资产多指标排名 |
| `composite_score()` | 加权复合评分 + 总排名 |

### OOP 算子类

每个功能模块同时提供类封装，继承 `TimeSeriesOperator` 基类：

```python
from operators.momentum import MomentumOperator

op = MomentumOperator()
result = op.fit_transform(return_series)
print(result["snr"], result["hurst"])
```

---

## 🏗️ 构建哲学

### 算子优先 (Operators-First)

每个分析函数都是**无状态、可测试**的纯函数，输入 Series → 输出指标。在此基础上提供 OOP 封装，兼顾函数式简洁与面向对象可组合性。

### 渐进式重构

项目采用"先建立结构，再填充内容"的策略。第一版建立完整的算子框架和测试体系，后续按模块逐步：

1. ✅ 算子库骨架 + 现有代码提炼
2. 🔄 `strategy/` 回测引擎完整化
3. ⏳ `volume.py` 量价算子实现
4. ⏳ 示例 Notebook 与策略报告

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
