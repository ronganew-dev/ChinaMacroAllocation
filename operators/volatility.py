"""
波动率/风险类算子 — 资产风险度量和波动特征计算

包含：
- calc_ewma_cov: 指数加权移动平均协方差矩阵
- calc_drawdowns: 最大回撤及相关指标
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from operators.base import TimeSeriesOperator


# ---------------------------------------------------------------------------
# EWMA 协方差矩阵
# ---------------------------------------------------------------------------
def calc_ewma_cov(
    data: np.ndarray,
    decay_factor: float,
) -> np.ndarray:
    """
    计算指数加权移动平均协方差矩阵。

    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_assets)
        收益率矩阵，每列为一个资产。
    decay_factor : float
        衰减因子 (0 < decay_factor < 1)，越小衰减越快。

    Returns
    -------
    np.ndarray, shape (n_assets, n_assets)
    """
    n = data.shape[0]
    weights = np.array(
        [(1 - decay_factor) * decay_factor ** i for i in range(n - 1, -1, -1)]
    )
    weights = weights / weights.sum()

    mean = np.dot(weights, data)
    centered = data - mean
    weighted_centered = centered * np.sqrt(weights.reshape(-1, 1))
    return np.dot(weighted_centered.T, weighted_centered)


# ---------------------------------------------------------------------------
# 回撤计算
# ---------------------------------------------------------------------------
def calc_drawdowns(
    cumu_value: np.ndarray,
    dates: pd.DatetimeIndex,
    is_compounding: bool = True,
) -> Dict[str, Any]:
    """
    计算最大回撤、峰值日期、低谷日期、恢复日期。

    Parameters
    ----------
    cumu_value : np.ndarray
        累计净值序列。
    dates : pd.DatetimeIndex
        对应日期索引。
    is_compounding : bool
        True → 复利净值（相对回撤）；False → 累加序列（绝对回撤）。

    Returns
    -------
    Dict with keys: peak, peak_date, trough_date, recovery_date
    """
    nav = pd.Series(cumu_value, index=dates)
    if is_compounding:
        nav = nav / nav.iloc[0]

    max_nav = nav.expanding().max()

    if is_compounding:
        safe_max = max_nav.replace(0, np.nan)
        drawdown = nav / safe_max - 1
    else:
        drawdown = nav - max_nav

    drawdown = drawdown.replace([np.inf, -np.inf], np.nan)
    max_dd = drawdown.min(skipna=True)
    peak_date = max_nav.idxmax()

    valid_dd = drawdown.dropna()
    if valid_dd.empty:
        return {
            "peak": [0.0],
            "peak_date": [dates[0]],
            "trough_date": [dates[0]],
            "recovery_date": [dates[0]],
        }

    trough_date = valid_dd.idxmin()
    recovery_date = None
    try:
        loc = drawdown.index.get_loc(trough_date)
        for i in range(loc, len(drawdown)):
            if drawdown.iloc[i] >= 0:
                recovery_date = drawdown.index[i]
                break
    except (KeyError, ValueError):
        pass

    return {
        "peak": [max_dd],
        "peak_date": [peak_date],
        "trough_date": [trough_date],
        "recovery_date": [recovery_date],
    }


# ===================================================================
# 算子类封装
# ===================================================================

class EWMAVolatilityOperator(TimeSeriesOperator):
    """EWMA 波动率协方差算子"""

    def __init__(self, decay_factor: float = 0.5 ** (1 / 63), name: str = None):
        super().__init__(name or "EWMAVolatilityOperator")
        self.decay_factor = decay_factor
        self._cov_matrix: Optional[np.ndarray] = None

    def fit(self, series: pd.Series) -> "EWMAVolatilityOperator":
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        # 单资产转列向量
        data = series.values.reshape(-1, 1)
        self._cov_matrix = calc_ewma_cov(data, self.decay_factor)
        return self._cov_matrix


class DrawdownOperator(TimeSeriesOperator):
    """回撤分析算子"""

    def __init__(self, is_compounding: bool = True, name: str = None):
        super().__init__(name or "DrawdownOperator")
        self.is_compounding = is_compounding
        self._result: Optional[Dict[str, Any]] = None

    def fit(self, series: pd.Series) -> "DrawdownOperator":
        return self

    def transform(self, series: pd.Series) -> Dict[str, Any]:
        # 需要 series 是累计净值，用 cumprod 转换
        cumu = (1 + series).cumprod().values
        self._result = calc_drawdowns(cumu, series.index, self.is_compounding)
        return self._result
