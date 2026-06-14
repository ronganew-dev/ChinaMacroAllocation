"""
动量/趋势类算子 — 用于检测资产价格的趋势特征

包含：
- compute_autocorr: 滞后自相关
- compute_snr: 信噪比 (年化均值 / 年化波动)
- compute_hurst: Hurst 指数 (R/S 分析法)
- compute_streaks: 连涨连跌极值
- compute_trend_efficiency: 趋势效率 + 趋势强度
- compute_trend_stability: 趋势稳定性 (均线方向一致性)
- compute_ma_backtest: 双均线交叉策略回测
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from operators.base import TimeSeriesOperator

BDAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 自相关
# ---------------------------------------------------------------------------
def compute_autocorr(series: pd.Series, lags: List[int] = None) -> Dict[str, float]:
    """
    计算序列的滞后自相关系数。

    Parameters
    ----------
    series : pd.Series
        日收益率序列。
    lags : List[int], optional
        滞后阶数列表，默认 [1, 5, 20]。

    Returns
    -------
    Dict[str, float]
        e.g. {"AC(1)": 0.05, "AC(5)": -0.02, "AC(20)": 0.01}
    """
    if lags is None:
        lags = [1, 5, 20]
    return {f"AC({lag})": series.autocorr(lag=lag) for lag in lags}


# ---------------------------------------------------------------------------
# 信噪比
# ---------------------------------------------------------------------------
def compute_snr(series: pd.Series, bdays_per_year: int = BDAYS_PER_YEAR) -> float:
    """
    信噪比 = 年化均值 / 年化波动率。

    越高表示趋势收益越突出，越适合趋势跟踪策略。
    """
    ann_mean = series.mean() * bdays_per_year
    ann_vol = series.std() * np.sqrt(bdays_per_year)
    return ann_mean / ann_vol if ann_vol > 0 else np.nan


# ---------------------------------------------------------------------------
# Hurst 指数
# ---------------------------------------------------------------------------
def compute_hurst(series: pd.Series, min_block: int = 4) -> float:
    """
    Hurst 指数 (R/S 重标极差分析)。

    > 0.5 → 趋势性，< 0.5 → 均值回归，= 0.5 → 随机游走。
    """
    n = len(series)
    if n < 100:
        return np.nan

    data = series.dropna().values
    max_block = n // 4
    block_sizes = np.unique(
        np.logspace(np.log10(min_block), np.log10(max_block), 50).astype(int)
    )

    rs_values, valid_sizes = [], []
    for m in block_sizes:
        if m >= n:
            break
        n_blocks = n // m
        rs_blocks = []
        for i in range(n_blocks):
            block = data[i * m : (i + 1) * m]
            deviations = block - block.mean()
            R = deviations.cumsum().max() - deviations.cumsum().min()
            S = block.std(ddof=1)
            if S > 0 and R > 0:
                rs_blocks.append(R / S)
        if rs_blocks:
            rs_values.append(np.mean(rs_blocks))
            valid_sizes.append(m)

    if len(rs_values) < 6:
        return np.nan

    slope, _ = np.polyfit(np.log(valid_sizes), np.log(rs_values), 1)
    return slope


# ---------------------------------------------------------------------------
# 趋势稳定性
# ---------------------------------------------------------------------------
def compute_trend_stability(
    series: pd.Series, fast: int = 20, slow: int = 60
) -> float:
    """
    趋势稳定性 — 快慢均线斜率方向一致的样本比例。

    衡量价格运动的方向一致性，越接近 1 越适合趋势跟踪。
    """
    price = (1 + series).cumprod()
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    slope_fast, slope_slow = sma_fast.diff(), sma_slow.diff()
    valid = ~(slope_fast.isna() | slope_slow.isna())
    if valid.sum() == 0:
        return np.nan
    consistent = np.sign(slope_fast[valid]) == np.sign(slope_slow[valid])
    return consistent.sum() / valid.sum()


# ---------------------------------------------------------------------------
# 连涨连跌极值
# ---------------------------------------------------------------------------
def compute_streaks(series: pd.Series) -> Dict[str, int]:
    """
    连涨/连跌记录。

    Returns
    -------
    {"max_consec_up": int, "max_consec_down": int}
    """
    pos_mask = (series > 0).values
    neg_mask = (series < 0).values

    max_pos = cur = 0
    for v in pos_mask:
        cur = cur + 1 if v else 0
        max_pos = max(max_pos, cur)

    max_neg = cur = 0
    for v in neg_mask:
        cur = cur + 1 if v else 0
        max_neg = max(max_neg, cur)

    return {"max_consec_up": max_pos, "max_consec_down": max_neg}


# ---------------------------------------------------------------------------
# 趋势效率 + 趋势强度
# ---------------------------------------------------------------------------
def compute_trend_efficiency(series: pd.Series) -> Dict[str, float]:
    """
    趋势效率 & 趋势强度。

    - **效率**：累计收益 / 绝对收益之和。接近 ±1 表示方向高度一致。
    - **强度**：累计收益率的绝对值，衡量趋势的"力度"。
    """
    cum_return = (1 + series).prod() - 1
    sum_abs = series.abs().sum()
    efficiency = cum_return / sum_abs if sum_abs > 0 else 0.0
    return {"trend_efficiency": efficiency, "trend_strength": abs(cum_return)}


# ---------------------------------------------------------------------------
# 双均线交叉回测
# ---------------------------------------------------------------------------
def compute_ma_backtest(
    series: pd.Series,
    fast: int = 20,
    slow: int = 60,
    bdays_per_year: int = BDAYS_PER_YEAR,
) -> Dict[str, Union[float, int]]:
    """
    双均线趋势跟踪策略回测。

    规则：前一日 fast_MA > slow_MA → 做多；反之做空。
    返回年化收益、波动、夏普、最大回撤、胜率、交易次数、Calmar。
    """
    price = (1 + series).cumprod()
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()

    # 前一日均线决定今日持仓（防未来偏移）
    signal = pd.Series(0, index=series.index, dtype=float)
    signal[sma_fast.shift(1) > sma_slow.shift(1)] = 1.0
    signal[sma_fast.shift(1) < sma_slow.shift(1)] = -1.0

    strat_returns = signal * series
    valid = signal != 0
    sr = strat_returns[valid]

    if len(sr) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "num_trades": np.nan,
            "calmar": np.nan,
        }

    nav = (1 + sr).cumprod()
    n_years = len(sr) / bdays_per_year
    ann_return = nav.iloc[-1] ** (1 / n_years) - 1
    ann_vol = sr.std() * np.sqrt(bdays_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    running_max = nav.expanding().max()
    dd = nav / running_max - 1
    max_dd = dd.min()
    win_rate = (sr > 0).sum() / len(sr)
    num_trades = int((signal.diff() != 0).sum() // 2)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "calmar": calmar,
    }


# ===================================================================
# 算子类封装 (OOP 风格，与函数式共存)
# ===================================================================

class MomentumOperator(TimeSeriesOperator):
    """动量趋势算子 — 封装多个趋势分析指标"""

    def __init__(self, name: str = "MomentumOperator"):
        super().__init__(name)
        self._result: Dict[str, Any] = {}

    def fit(self, series: pd.Series) -> "MomentumOperator":
        return self

    def transform(self, series: pd.Series) -> Dict[str, Any]:
        self._result = {
            "snr": compute_snr(series),
            "hurst": compute_hurst(series),
            "autocorr": compute_autocorr(series),
            "streaks": compute_streaks(series),
            "trend_efficiency": compute_trend_efficiency(series),
            "trend_stability": compute_trend_stability(series),
        }
        return self._result
