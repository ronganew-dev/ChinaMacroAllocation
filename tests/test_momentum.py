"""
测试 — 动量趋势算子 (Formulaic Operators)

验证所有函数为纯向量化、无状态调用。
"""

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def random_walk():
    """模拟随机游走收益率 (H ≈ 0.5)"""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    return pd.Series(returns)


@pytest.fixture
def trending_series():
    """强趋势序列 (持续小正收益)"""
    np.random.seed(42)
    n = 500
    base = np.full(n, 0.002)
    noise = np.random.randn(n) * 0.005
    return pd.Series(base + noise)


@pytest.fixture
def mean_reverting():
    """均值回归序列 (负自相关)"""
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.randn(n) * 0.02)
    return returns * -0.3


# ---------------------------------------------------------------------------
# 信噪比 | SNR
# ---------------------------------------------------------------------------
def test_snr_positive_for_trending(trending_series):
    """趋势序列应有正信噪比"""
    v = snr(trending_series)
    assert v > 0, f"Expected positive SNR, got {v}"


def test_snr_returns_float(random_walk):
    v = snr(random_walk)
    assert isinstance(v, float)
    assert not np.isnan(v)


# ---------------------------------------------------------------------------
# Hurst 指数 | Hurst Exponent
# ---------------------------------------------------------------------------
def test_hurst_trending_above_half(trending_series):
    """趋势序列 Hurst > 0.5"""
    h = hurst_exponent(trending_series)
    assert h > 0.5, f"Expected H > 0.5 for trending series, got {h}"


def test_hurst_random_walk_approx_half(random_walk):
    """随机游走 Hurst ≈ 0.5"""
    h = hurst_exponent(random_walk)
    assert 0.35 < h < 0.65, f"Expected H ≈ 0.5, got {h}"


# ---------------------------------------------------------------------------
# 自相关 | ACF
# ---------------------------------------------------------------------------
def test_acf_returns_float(random_walk):
    v = acf(random_walk, lag=1)
    assert isinstance(v, float)
    assert not np.isnan(v)


def test_acf_multi_returns_dict(random_walk):
    result = acf_multi(random_walk)
    assert isinstance(result, dict)
    assert "AC(1)" in result
    assert "AC(5)" in result
    assert "AC(20)" in result


def test_acf_multi_custom_lags(random_walk):
    result = acf_multi(random_walk, lags=[1, 10])
    assert "AC(10)" in result
    assert "AC(20)" not in result


# ---------------------------------------------------------------------------
# 连涨连跌 | Streaks
# ---------------------------------------------------------------------------
def test_streaks_returns_dict(trending_series):
    s = streaks(trending_series)
    assert "max_consec_up" in s
    assert "max_consec_down" in s
    assert s["max_consec_up"] >= 0
    assert s["max_consec_down"] >= 0


def test_streaks_all_positive():
    """全正收益序列"""
    s = streaks(pd.Series([0.01] * 10))
    assert s["max_consec_up"] == 10
    assert s["max_consec_down"] == 0


def test_streaks_all_negative():
    """全负收益序列"""
    s = streaks(pd.Series([-0.01] * 10))
    assert s["max_consec_up"] == 0
    assert s["max_consec_down"] == 10


# ---------------------------------------------------------------------------
# 趋势效率 | Trend Efficiency
# ---------------------------------------------------------------------------
def test_trend_efficiency_perfect():
    """完美单调上涨 → 效率接近 1"""
    s = pd.Series([0.001] * 100)
    eff = trend_efficiency(s)
    assert abs(eff) > 0.9, f"Expected efficiency near 1, got {eff}"


def test_trend_efficiency_noisy(random_walk):
    """随机游走效率应接近 0"""
    eff = trend_efficiency(random_walk)
    assert -0.5 < eff < 0.5


# ---------------------------------------------------------------------------
# 趋势强度 | Trend Strength
# ---------------------------------------------------------------------------
def test_trend_strength_positive(trending_series):
    v = trend_strength(trending_series)
    assert v > 0


def test_trend_strength_zero_for_flat(random_walk):
    v = trend_strength(pd.Series(np.zeros(100)))
    assert v == 0.0


# ---------------------------------------------------------------------------
# 趋势稳定性 | Trend Stability
# ---------------------------------------------------------------------------
def test_trend_stability_range(trending_series):
    v = trend_stability(trending_series)
    assert 0 <= v <= 1


# ---------------------------------------------------------------------------
# 双均线回测 | MA Crossover
# ---------------------------------------------------------------------------
def test_ma_crossover_returns_dict(trending_series):
    r = ma_crossover_metrics(trending_series)
    assert "ann_return" in r
    assert "sharpe" in r
    assert "max_drawdown" in r
    assert "win_rate" in r
    assert "num_trades" in r


def test_ma_crossover_trending_profitable(trending_series):
    """趋势序列上均线策略应盈利"""
    r = ma_crossover_metrics(trending_series)
    assert r["ann_return"] > 0, f"Expected positive return, got {r['ann_return']}"
    assert r["sharpe"] > 0, f"Expected positive sharpe, got {r['sharpe']}"
