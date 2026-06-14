"""
波动率/风险类算子 — Formulaic Operators

每个函数 = 一个原子公式，纯向量化，零显式循环。
输入 pd.Series / np.ndarray → 输出 np.ndarray / Dict。

参考标准：WorldQuant BRAIN Formulaic Operators 工程规范。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ===================================================================
# EWMA 协方差矩阵 | Exponentially Weighted Covariance
# ===================================================================

def ewma_weights(n: int, decay: float) -> np.ndarray:
    """
    生成 EWMA 衰减权重向量 — 纯向量化，无显式循环。

    Formula:
        w_i = (1 - λ) · λ^{n-1-i}   for i = 0, ..., n-1

    Parameters
    ----------
    n : int
        样本长度。
    decay : float
        衰减因子 λ (0 < λ < 1)，越小旧数据衰减越快。

    Returns
    -------
    np.ndarray, shape (n,)
    """
    # 向量化生成指数序列: λ^{n-1}, λ^{n-2}, ..., λ^0
    powers = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = (1 - decay) * (decay ** powers)  # 全向量化
    return weights / weights.sum()


def ewma_cov(data: np.ndarray, decay: float = 0.94) -> np.ndarray:
    r"""
    指数加权移动平均协方差矩阵 — 全程向量化。

    Formula:
        Σ_EWMA = Σ_i w_i · (x_i - μ_w)^T · (x_i - μ_w)

    其中 w_i 为指数衰减权重，μ_w 为加权均值。

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
        T 个时间点、N 个资产的收益率矩阵。
    decay : float
        衰减因子 λ，默认 0.94（日频 EWMA 标准值）。

    Returns
    -------
    np.ndarray, shape (N, N)
        协方差矩阵。
    """
    T = data.shape[0]
    w = ewma_weights(T, decay)  # (T,), 纯向量化

    # 加权均值
    mean = np.dot(w, data)  # (N,)

    # 中心化 + 加权
    centered = data - mean  # (T, N)
    weighted = centered * np.sqrt(w[:, np.newaxis])  # (T, N)

    return weighted.T @ weighted  # (N, N)


# ===================================================================
# 最大回撤 | Maximum Drawdown
# ===================================================================

def drawdown_series(nav: pd.Series) -> pd.Series:
    r"""
    回撤序列 — 纯向量化。

    Formula:
        DD(t) = NAV(t) / max_{s ≤ t} NAV(s) - 1

    Parameters
    ----------
    nav : pd.Series
        累计净值序列。

    Returns
    -------
    pd.Series，对齐 nav 的索引。
    """
    running_max = nav.expanding().max()
    safe = running_max.replace(0, np.nan)
    dd = nav / safe - 1
    return dd.replace([np.inf, -np.inf], np.nan)


def max_drawdown(nav: pd.Series) -> float:
    r"""
    最大回撤。

    Formula:
        MaxDD = min_t [ NAV(t) / max_{s ≤ t} NAV(s) - 1 ]
    """
    dd = drawdown_series(nav)
    return float(dd.min(skipna=True))


def drawdown_details(nav: pd.Series) -> Dict[str, Any]:
    """
    最大回撤详情 — 峰值、谷值、恢复时间。

    全程向量化，零显式循环。

    Returns
    -------
    Dict with keys:
        peak_value, peak_date, trough_value, trough_date,
        recovery_date, max_drawdown
    """
    running_max = nav.expanding().max()
    dd = nav / running_max.replace(0, np.nan) - 1
    dd = dd.replace([np.inf, -np.inf], np.nan)

    if dd.dropna().empty:
        return {
            "peak_value": float(nav.iloc[0]),
            "peak_date": nav.index[0],
            "trough_value": float(nav.iloc[0]),
            "trough_date": nav.index[0],
            "recovery_date": nav.index[0],
            "max_drawdown": 0.0,
        }

    trough_idx = dd.idxmin()
    trough_val = float(nav.loc[trough_idx])
    max_dd = float(dd.loc[trough_idx])

    # 峰值 = 回撤起点（running_max 为最高值时的日期）
    peak_date = running_max.loc[:trough_idx].idxmax()
    peak_val = float(nav.loc[peak_date])

    # 恢复日期：谷底后首次回到峰值（全向量化）
    after_trough = nav.index > trough_idx
    if after_trough.any():
        recovered = (nav >= peak_val) & after_trough
        recovery_date = nav.index[recovered].min() if recovered.any() else None
    else:
        recovery_date = None

    return {
        "peak_value": peak_val,
        "peak_date": peak_date,
        "trough_value": trough_val,
        "trough_date": trough_idx,
        "recovery_date": recovery_date,
        "max_drawdown": max_dd,
    }


def recovery_time(nav: pd.Series) -> Optional[int]:
    """
    回撤恢复所需交易日数。

    返回从谷底到首次恢复的交易天数，若未恢复则返回 None。
    """
    info = drawdown_details(nav)
    if info["recovery_date"] is None:
        return None
    loc = nav.index.get_loc(info["trough_date"])
    if loc is None:
        return None
    loc_rec = nav.index.get_loc(info["recovery_date"])
    return int(loc_rec - loc)


def ulcer_index(nav: pd.Series) -> float:
    r"""
    Ulcer 指数 — 衡量回撤深度与持续时间的综合指标。

    Formula:
        UI = sqrt( (1/N) · Σ_t DD(t)^2 )
    """
    dd = drawdown_series(nav)
    return float(np.sqrt((dd**2).mean(skipna=True)))
