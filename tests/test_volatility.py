"""
测试 — 波动率/风险算子 (Formulaic Operators)

验证所有函数为纯向量化、无状态调用。
"""

import numpy as np
import pandas as pd
import pytest

from operators.volatility import (
    ewma_cov,
    ewma_weights,
    drawdown_series,
    max_drawdown,
    drawdown_details,
    recovery_time,
    ulcer_index,
)


# ---------------------------------------------------------------------------
# EWMA 权重 | EWMA Weights
# ---------------------------------------------------------------------------
def test_ewma_weights_shape():
    w = ewma_weights(100, 0.94)
    assert w.shape == (100,)
    assert abs(w.sum() - 1.0) < 1e-10


def test_ewma_weights_decay():
    """近期权重应大于远期权重"""
    w = ewma_weights(50, 0.94)
    assert w[0] < w[-1]  # 第一个（最远）< 最后一个（最近）


# ---------------------------------------------------------------------------
# EWMA 协方差 | EWMA Covariance
# ---------------------------------------------------------------------------
def test_ewma_cov_shape():
    """2 资产 100 天 → 2×2 矩阵"""
    np.random.seed(42)
    data = np.random.randn(100, 2) * 0.01
    cov = ewma_cov(data, 0.94)
    assert cov.shape == (2, 2)
    assert cov[0, 1] == pytest.approx(cov[1, 0], abs=1e-12)


def test_ewma_cov_positive_diagonal():
    """对角线应为正数 (方差)"""
    data = np.random.randn(50, 3) * 0.02
    cov = ewma_cov(data, 0.94)
    assert all(cov[i, i] > 0 for i in range(3))


# ---------------------------------------------------------------------------
# 回撤序列 | Drawdown Series
# ---------------------------------------------------------------------------
def test_drawdown_series_values():
    nav = pd.Series([1.0, 0.95, 0.90, 0.85, 0.80])
    dd = drawdown_series(nav)
    assert dd.iloc[0] == 0.0  # 第一天回撤 = 0
    assert dd.iloc[-1] < 0  # 连续下跌回撤为负


# ---------------------------------------------------------------------------
# 最大回撤 | Max Drawdown
# ---------------------------------------------------------------------------
def test_max_drawdown_negative():
    nav = pd.Series([1.0, 0.95, 0.90, 0.85, 0.80])
    mdd = max_drawdown(nav)
    assert mdd < 0


def test_max_drawdown_stable():
    nav = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20])
    mdd = max_drawdown(nav)
    assert mdd >= 0  # 持续上涨无回撤


# ---------------------------------------------------------------------------
# 回撤详情 | Drawdown Details
# ---------------------------------------------------------------------------
def test_drawdown_details_recovery():
    """下跌后恢复 → recovery_date 不为 None"""
    nav = pd.Series([1.0, 0.9, 0.85, 0.95, 1.05, 1.1])
    details = drawdown_details(nav)
    assert details["recovery_date"] is not None
    assert details["max_drawdown"] < 0


def test_drawdown_details_no_decline():
    """持续上涨 → max_drawdown >= 0"""
    nav = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20])
    details = drawdown_details(nav)
    assert details["max_drawdown"] >= 0


def test_drawdown_details_keys():
    nav = pd.Series(np.random.randn(200) * 0.01)
    nav = (1 + nav).cumprod()
    details = drawdown_details(nav)
    assert "peak_value" in details
    assert "peak_date" in details
    assert "trough_date" in details
    assert "recovery_date" in details
    assert "max_drawdown" in details


# ---------------------------------------------------------------------------
# 恢复时间 | Recovery Time
# ---------------------------------------------------------------------------
def test_recovery_time_positive():
    nav = pd.Series([1.0, 0.9, 0.85, 0.95, 1.05, 1.1])
    t = recovery_time(nav)
    assert t is not None and t >= 0


def test_recovery_time_none():
    """未恢复 → None"""
    nav = pd.Series([1.0, 0.9, 0.85, 0.83])
    t = recovery_time(nav)
    assert t is None


# ---------------------------------------------------------------------------
# Ulcer Index
# ---------------------------------------------------------------------------
def test_ulcer_index_non_negative():
    nav = pd.Series([1.0, 0.95, 0.98, 0.92, 1.05])
    ui = ulcer_index(nav)
    assert ui >= 0


def test_ulcer_index_zero_for_rising():
    nav = pd.Series(np.linspace(1.0, 2.0, 100))
    ui = ulcer_index(nav)
    assert ui == pytest.approx(0.0, abs=1e-10)
