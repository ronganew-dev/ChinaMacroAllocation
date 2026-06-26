# ChinaMacroAllocation — 通用量化算子库

> 从实盘策略中提炼的开箱即用**量化小工具集合**，可独立复用于趋势跟踪、动量分析、波动率建模、周期识别、横截面评分等场景。
>
> 所有算子均为纯函数式、全程向量化、原子粒度。每个函数就是一个公式，拿来即用。

---

## 快速开始

```bash
pip install -r requirements.txt
```

```python
from operators import snr, hurst_exponent, drawdown_details

# 趋势强度
signal = snr(return_series)

# 趋势持续性
h = hurst_exponent(return_series)

# 回撤分析
dd = drawdown_details((1 + return_series).cumprod())
print(dd["max_drawdown"], dd["recovery_date"])
```

---

## 算子一览

### 动量 / 趋势 (`momentum.py`)

| 函数 | 说明 |
|------|------|
| `acf()` / `acf_multi()` | 单阶 / 多阶滞后自相关 |
| `snr()` | 信噪比（年化收益 / 年化波动） |
| `hurst_exponent()` | Hurst 指数（R/S 重标极差，纯向量化） |
| `streaks()` | 连涨连跌极值 |
| `trend_efficiency()` | 趋势效率：总收益 / 总振幅 |
| `trend_strength()` | 趋势强度：累计净值偏离 |
| `trend_stability()` | 趋势稳定性：快慢均线同向比例 |
| `ma_crossover_metrics()` | 均线交叉策略绩效 |

### 波动率 / 风险 (`volatility.py`)

| 函数 | 说明 |
|------|------|
| `ewma_weights()` | EWMA 衰减权重 |
| `ewma_cov()` | 指数加权协方差矩阵 |
| `drawdown_series()` / `max_drawdown()` | 回撤序列 / 最大回撤 |
| `drawdown_details()` | 回撤峰值、谷底、恢复详情 |
| `recovery_time()` | 恢复所需交易日数 |
| `ulcer_index()` | Ulcer 指数 |

### 横截面 (`cross_sectional.py`)

| 函数 | 说明 |
|------|------|
| `cross_sectional_rank()` | 多资产多指标排名（方向可配） |
| `cross_sectional_zscore()` | 横截面 Z-Score 标准化 |
| `composite_score()` | 加权复合评分（矩阵向量化） |
| `sector_neutralize()` | 行业中性化（去均值 / 标准化） |

### 周期分析 (`fourier_cycle.py`, `sin_fit.py`, `HP_filter.py`)

| 函数 | 说明 |
|------|------|
| FFT 频谱分析 | 提取序列主导周期 |
| 正弦波拟合 | 非线性最小二乘拟合相位/振幅 |
| HP 滤波 | 趋势-周期分解 |

### 宏观数据 (`macro_data_fetcher.py`, `China_macro_data_cleanser.py`)

| 函数 | 说明 |
|------|------|
| 数据获取 | FRED / Wind 宏观指标拉取 |
| 数据清洗 | 异常值处理、频次对齐、插值填充 |

### 辅助工具

| 模块 | 说明 |
|------|------|
| `calendar_utils.py` | 交易日历、偏移计算 |
| `risk_control.py` | 风控规则引擎 |
| `BryBroschan.py` | Bry-Broschan 周期拐点识别 |
| `OECD_synthetic.py` | OECD 合成指数构建 |

---

## 构建哲学 — Formulaic Operators

1. **纯函数式** — 没有类、没有 `fit/transform`、没有内部状态。每个算子就是一个纯函数：`f(Series) → value`
2. **全程向量化** — 所有算子通过 pandas / numpy 向量化实现，零显式 Python 数据元素循环
3. **原子粒度** — 每个函数只完成一个公式的计算
4. **pandas-first** — 时序算子直接接受 `pd.Series` 作为输入，保持索引对齐

结果：**可单元测试、可组合、可复用**。

---

## 技术栈

| 组件 | 选型 |
|------|------|
| 数据处理 | pandas, numpy |
| 统计分析 | scipy |
| 测试 | pytest |

---

## 许可

MIT License
