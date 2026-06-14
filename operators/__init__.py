"""
ChinaMacroAllocation — Formulaic Operators 通用量化算子库

工程规范：
- 纯函数式，无类、无状态、无 fit/transform
- 全程向量化（pandas / numpy），零显式数据元素循环
- 原子粒度：每个函数 = 一个公式
- 输入/输出全部为 pandas Series / DataFrame

参考标准：WorldQuant BRAIN Formulaic Operators
"""

# ── momentum ──
from operators.momentum import (
    acf,
    acf_multi,
    snr,
    hurst_exponent,
    streaks,
    trend_efficiency,
    trend_strength,
    trend_stability,
    ma_crossover_metrics,
)

# ── volatility ──
from operators.volatility import (
    ewma_cov,
    ewma_weights,
    drawdown_series,
    max_drawdown,
    drawdown_details,
    recovery_time,
    ulcer_index,
)

# ── cross_sectional ──
from operators.cross_sectional import (
    cross_sectional_rank,
    cross_sectional_zscore,
    composite_score,
    sector_neutralize,
)

# ── base utility ──
from operators.base import nanmask

__all__ = [
    # momentum
    "acf", "acf_multi", "snr", "hurst_exponent",
    "streaks", "trend_efficiency", "trend_strength",
    "trend_stability", "ma_crossover_metrics",
    # volatility
    "ewma_cov", "ewma_weights",
    "drawdown_series", "max_drawdown", "drawdown_details",
    "recovery_time", "ulcer_index",
    # cross_sectional
    "cross_sectional_rank", "cross_sectional_zscore",
    "composite_score", "sector_neutralize",
    # utility
    "nanmask",
]
