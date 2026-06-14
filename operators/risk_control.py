"""
风险控制算子 — Formulaic Operators

时序动量评估 + 估值百分位 + 风险预算乘数。

工程规范（WorldQuant BRAIN Formulaic Operators）：
- 纯函数式，无类、无状态
- 全程向量化，零显式 Python 数据元素循环
- 原子粒度：每个函数 = 一个公式
- pandas-first：输入/输出全部为 pd.Series / np.ndarray
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from operators.base import nanmask

# ===================================================================
# 时序动量信号 | Time-Series Momentum Signal
# ===================================================================

def tsmom_signal(
    price: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    r"""
    经典时序动量信号 — Moskowitz, Ooi & Pedersen (2012)。

    Formula:
        signal_t = sign( price_t / price_{t-lookback} - 1 )

    Parameters
    ----------
    price : pd.Series
        价格 / 净值序列（非收益率）。
    lookback : int
        回溯期，默认 252 个交易日（≈12个月）。

    Returns
    -------
    pd.Series
        信号值：1.0 (多头) / 0.0 (中性) / -1.0 (空头)。
    """
    trailing_ret = price / price.shift(lookback) - 1.0
    signal = pd.Series(np.sign(trailing_ret), index=price.index)
    signal = signal.replace(0.0, np.nan).ffill().fillna(0.0)
    return signal


def tsmom_strength(
    price: pd.Series,
    lookback: int = 252,
    vol_lookback: int = 63,
) -> pd.Series:
    r"""
    时序动量强度 — 经波动率调整后的连续动量度量。

    Formula:
        strength_t = r_{t-lookback:t} / σ_{t-vol_lookback:t}

    信号平滑化版本，取值越大表示趋势越明确。

    Parameters
    ----------
    price : pd.Series
        价格序列。
    lookback : int
        收益回溯期，默认 252。
    vol_lookback : int
        波动率回溯期，默认 63（≈3个月）。

    Returns
    -------
    pd.Series
        连续动量强度，可正可负。
    """
    returns = price.pct_change().dropna()
    trailing_ret = price / price.shift(lookback) - 1.0
    vol = returns.rolling(vol_lookback).std() * np.sqrt(252)

    strength = trailing_ret / vol.replace(0, np.nan)
    return strength


def tsmom_ma_signal(
    price: pd.Series,
    fast: int = 20,
    slow: int = 60,
) -> pd.Series:
    r"""
    均线相对动量信号 — 价格相对于移动平均线的位置。

    Formula:
        signal_t = ( price_t / SMA(price, slow)_t - 1 ) · scale

    正值 → 价格在均线上方（顺势），负值 → 在均线下方（逆势）。
    适用于"跌破均线则减配"场景。

    Parameters
    ----------
    price : pd.Series
        价格序列。
    fast : int
        快线窗口，默认 20。
    slow : int
        慢线窗口，默认 60（用于判断趋势方向）。

    Returns
    -------
    pd.Series
        [-1, 1] 之间的连续信号。
    """
    sma_slow = price.rolling(slow).mean()
    ratio = price / sma_slow - 1.0
    # 用快线 SMA 的波动来 scale，使信号对低波动时期更敏感
    sma_fast = price.rolling(fast).mean()
    ma_vol = sma_fast.pct_change().rolling(slow).std()

    scaled = ratio / ma_vol.replace(0, np.nan)
    # 裁剪到 [-1, 1] 区间
    return scaled.clip(-1.0, 1.0)


def tsmom_signal_binary(
    price: pd.Series,
    slow: int = 60,
    threshold: float = 0.0,
) -> pd.Series:
    r"""
    均线二元信号 — 价格是否在均线上方。

    Formula:
        signal_t = 1.0   if price_t / SMA(price, slow)_t > 1 + threshold
                   0.0   otherwise

    适用于直接作为"做多/减仓"决策标志。

    Parameters
    ----------
    price : pd.Series
    slow : int
        均线窗口，默认 60。
    threshold : float
        阈值偏移，默认 0.0（价格刚好在均线上方即做多）。

    Returns
    -------
    pd.Series
        取值 {0.0, 1.0}。
    """
    sma = price.rolling(slow).mean()
    above = (price / sma > 1.0 + threshold).astype(float)
    return above


# ===================================================================
# 估值百分位 | Valuation Percentile
# ===================================================================

def valuation_percentile(
    price: pd.Series,
    window: Optional[int] = None,
    method: str = "minmax",
) -> pd.Series:
    r"""
    估值百分位 — 当前价格在历史价格中的相对位置。

    Formula (method='minmax'):
        pct_t = ( price_t - min_{history} ) / ( max_{history} - min_{history} )

    Formula (method='rank'):
        pct_t = rank(price_t) / N

    Parameters
    ----------
    price : pd.Series
        价格序列。
    window : int, optional
        滚动窗口。None = 扩容窗口（使用全部历史）。
    method : str
        "minmax" → 最小-最大归一化
        "rank"   → 分位数排名

    Returns
    -------
    pd.Series
        [0, 1] 区间，值越大表示相对历史越贵。
    """
    if method == "minmax":
        if window is None:
            # 扩容窗口
            rolling_min = price.expanding().min()
            rolling_max = price.expanding().max()
        else:
            rolling_min = price.rolling(window).min()
            rolling_max = price.rolling(window).max()

        denom = rolling_max - rolling_min
        denom = denom.replace(0, np.nan)
        pct = (price - rolling_min) / denom

    elif method == "rank":
        if window is None:
            # 扩容窗口：当前值的百分位排名
            def _expanding_pct(series):
                return series.rank(pct=True).iloc[-1]

            pct = price.expanding().apply(
                _expanding_pct, raw=False
            )
        else:
            pct = price.rolling(window).apply(
                lambda s: s.rank(pct=True).iloc[-1], raw=False
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    return pct.clip(0.0, 1.0)


def valuation_zscore(
    price: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    r"""
    估值 Z-Score — 当前价格偏离历史均值的标准差倍数。

    Formula:
        z_t = ( price_t - μ_{history} ) / σ_{history}

    Parameters
    ----------
    price : pd.Series
    window : int, optional
        None = 扩容窗口。

    Returns
    -------
    pd.Series
        正 = 高于历史均值（偏贵），负 = 低于历史均值（便宜）。
    """
    if window is None:
        mu = price.expanding().mean()
        sigma = price.expanding().std()
    else:
        mu = price.rolling(window).mean()
        sigma = price.rolling(window).std()

    sigma = sigma.replace(0, np.nan)
    return (price - mu) / sigma


# ===================================================================
# 风险预算乘数 | Risk Budget Multiplier
# ===================================================================

def risk_multiplier(
    tsmom_signal: Union[pd.Series, np.ndarray],
    val_percentile: Union[pd.Series, np.ndarray],
    tsmom_weight: float = 0.5,
    val_threshold: float = 0.90,
    method: str = "multiplicative",
) -> pd.Series:
    r"""
    风险预算乘数 — 结合时序动量和估值，输出 [0, 1] 动态调整系数。

    Formula (method='multiplicative', default):
        m_t = M_t · (1 - P_t_penalty)

    Formula (method='additive'):
        m_t = w · M_t + (1-w) · (1 - P_t_penalty)

    其中：
        M_t          = max(tsmom, 0)  将方向信号映射到 [0, 1]
        P_t_penalty  = max(0, (percentile - threshold) / (1 - threshold))
        tsmom_weight = w

    Parameters
    ----------
    tsmom_signal : pd.Series or np.ndarray
        时序动量信号（可正可负，或已在 [0, 1]）。
    val_percentile : pd.Series or np.ndarray
        估值百分位 [0, 1]。
    tsmom_weight : float
        动量权重，默认 0.5。
    val_threshold : float
        估值触发阈值，默认 0.90（历史 90% 分位数）。
    method : str
        "multiplicative"（默认）或 "additive"。

    Returns
    -------
    pd.Series
        [0, 1] 之间的风险预算乘数，值越小 → 越应降低配置。
    """
    tsmom = np.asarray(tsmom_signal, dtype=np.float64)
    val = np.asarray(val_percentile, dtype=np.float64)

    # 将 tsmom 映射到 [0, 1]：负数 → 0，正数 → 原值（最大钳位到 1）
    M = np.clip(tsmom, 0.0, 1.0)

    # 估值惩罚：超过阈值后才开始线性增加
    P_penalty = np.clip((val - val_threshold) / (1.0 - val_threshold + 1e-10), 0.0, 1.0)

    if method == "multiplicative":
        multiplier = M * (1.0 - P_penalty)
    elif method == "additive":
        multiplier = tsmom_weight * M + (1.0 - tsmom_weight) * (1.0 - P_penalty)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = pd.Series(np.clip(multiplier, 0.0, 1.0))

    # 尝试保留索引
    if isinstance(tsmom_signal, pd.Series):
        result.index = tsmom_signal.index
    elif isinstance(val_percentile, pd.Series):
        result.index = val_percentile.index

    return result


def risk_status(
    multiplier: Union[pd.Series, np.ndarray, float],
    low_threshold: float = 0.3,
    mid_threshold: float = 0.7,
) -> pd.Series:
    r"""
    风险状态分类 — 将风险乘数映射为人可读的状态标签。

    Formula:
        status = "caution"   if multiplier < low_threshold
                 "watch"     if low_threshold <= multiplier < mid_threshold
                 "normal"    otherwise

    Parameters
    ----------
    multiplier : pd.Series, np.ndarray, or float
    low_threshold : float
        低风险阈值，默认 0.3。
    mid_threshold : float
        中风险阈值，默认 0.7。

    Returns
    -------
    pd.Series of str
    """
    m = np.asarray(multiplier, dtype=np.float64)
    cond_caution = m < low_threshold
    cond_watch = (m >= low_threshold) & (m < mid_threshold)

    labels = np.full_like(m, "normal", dtype=object)
    labels[cond_caution] = "caution"
    labels[cond_watch] = "watch"

    result = pd.Series(labels)
    if isinstance(multiplier, pd.Series):
        result.index = multiplier.index

    return result
