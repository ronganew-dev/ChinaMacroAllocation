"""
测试 — 波动率/风险算子
"""

import numpy as np
import pandas as pd
import pytest

from operators.volatility import (
    EWMAVolatilityOperator,
    DrawdownOperator,
    calc_drawdowns,
    calc_ewma_cov,
)


# ---------------------------------------------------------------------------
# EWMA 协方差
# ---------------------------------------------------------------------------
def test_ewma_cov_shape():
    """2 资产 100 天 → 2×2 矩阵"""
    np.random.seed(42)
    data = np.random.randn(100, 2) * 0.01
    decay = 0.94
    cov = calc_ewma_cov(data, decay)
    assert cov.shape == (2, 2)
    assert cov[0, 1] == cov[1, 0]  # 对称性


def test_ewma_cov_positive_diagonal():
    """对角线应为正数 (方差)"""
    data = np.random.randn(50, 3) * 0.02
    cov = calc_ewma_cov(data, 0.94)
    assert all(cov[i, i] > 0 for i in range(3))


def test_ewma_volatility_operator():
    op = EWMAVolatilityOperator(decay_factor=0.94)
    series = pd.Series(np.random.randn(100) * 0.01)
    result = op.fit_transform(series)
    assert result.shape == (1, 1)
    assert result[0, 0] > 0


# ---------------------------------------------------------------------------
# 回撤
# ---------------------------------------------------------------------------
def test_drawdowns_compounding():
    """复利净值：连续下跌应有负回撤"""
    nav = np.array([1.0, 0.95, 0.90, 0.85, 0.80])
    dates = pd.date_range("2020-01-01", periods=5)
    result = calc_drawdowns(nav, dates, is_compounding=True)
    assert result["peak"][0] < 0  # 最大回撤为负
    assert result["peak_date"][0] == dates[0]


def test_drawdowns_recovery():
    """下跌后恢复 → recovery_date 不为 None"""
    nav = np.array([1.0, 0.9, 0.85, 0.95, 1.05, 1.1])
    dates = pd.date_range("2020-01-01", periods=6)
    result = calc_drawdowns(nav, dates, is_compounding=True)
    assert result["recovery_date"][0] is not None


def test_drawdowns_no_decline():
    """持续上涨 → 无回撤"""
    nav = np.array([1.0, 1.05, 1.10, 1.15, 1.20])
    dates = pd.date_range("2020-01-01", periods=5)
    result = calc_drawdowns(nav, dates, is_compounding=True)
    assert result["peak"][0] >= 0  # 最大回撤 ≥ 0


def test_drawdown_operator():
    op = DrawdownOperator(is_compounding=True)
    series = pd.Series(np.random.randn(200) * 0.01)
    result = op.fit_transform(series)
    assert "peak" in result
    assert "peak_date" in result
    assert "trough_date" in result
    assert "recovery_date" in result
