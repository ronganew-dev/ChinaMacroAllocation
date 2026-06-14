"""
ChinaMacroAllocation — 通用量化算子库 (Quant Operators)

提供可复用的金融数据计算算子，按功能模块分组：
- momentum: 动量/趋势类 (自相关、信噪比、Hurst指数、连涨连跌等)
- volatility: 波动率/风险类 (EWMA协方差、回撤计算)
- cross_sectional: 横截面类 (排名、评分、中性化)
- volume: 量价结合类 (VWAP、资金流向等) — 🔧 开发中
"""

from operators.base import TimeSeriesOperator
from operators.momentum import (
    compute_autocorr,
    compute_snr,
    compute_hurst,
    compute_streaks,
    compute_trend_efficiency,
    compute_trend_stability,
    compute_ma_backtest,
)
from operators.volatility import calc_ewma_cov, calc_drawdowns
from operators.cross_sectional import cross_sectional_rank, composite_score

__all__ = [
    "TimeSeriesOperator",
    "compute_autocorr",
    "compute_snr",
    "compute_hurst",
    "compute_streaks",
    "compute_trend_efficiency",
    "compute_trend_stability",
    "compute_ma_backtest",
    "calc_ewma_cov",
    "calc_drawdowns",
    "cross_sectional_rank",
    "composite_score",
]
