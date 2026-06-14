"""
测试 — 动量趋势算子
"""

import numpy as np
import pandas as pd
import pytest

from operators.momentum import (
    MomentumOperator,
    compute_autocorr,
    compute_hurst,
    compute_ma_backtest,
    compute_snr,
    compute_streaks,
    compute_trend_efficiency,
    compute_trend_stability,
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
    base = np.full(n, 0.002)  # 每天 +0.2%
    noise = np.random.randn(n) * 0.005
    return pd.Series(base + noise)


@pytest.fixture
def mean_reverting():
    """均值回归序列 (震荡)"""
    np.random.seed(42)
    n = 500
    prices = np.cumsum(np.random.randn(n) * 0.02)
    # 取每日变化来模拟均值回归特征
    returns = pd.Series(np.diff(prices, prepend=0))
    return returns * -0.3  # 反向拉动 → 负自相关


# ---------------------------------------------------------------------------
# Tests: compute_snr
# ---------------------------------------------------------------------------
def test_snr_positive_for_trending(trending_series):
    """趋势序列应有正信噪比"""
    snr = compute_snr(trending_series)
    assert snr > 0, f"Expected positive SNR, got {snr}"


def test_snr_returns_float(random_walk):
    snr = compute_snr(random_walk)
    assert isinstance(snr, float)
    assert not np.isnan(snr)


# ---------------------------------------------------------------------------
# Tests: compute_hurst
# ---------------------------------------------------------------------------
def test_hurst_trending_above_half(trending_series):
    """趋势序列 Hurst > 0.5"""
    h = compute_hurst(trending_series)
    assert h > 0.5, f"Expected H > 0.5 for trending series, got {h}"


def test_hurst_random_walk_approx_half(random_walk):
    """随机游走 Hurst ≈ 0.5"""
    h = compute_hurst(random_walk)
    assert 0.35 < h < 0.65, f"Expected H ≈ 0.5, got {h}"


# ---------------------------------------------------------------------------
# Tests: compute_autocorr
# ---------------------------------------------------------------------------
def test_autocorr_returns_dict(random_walk):
    result = compute_autocorr(random_walk)
    assert isinstance(result, dict)
    assert "AC(1)" in result
    assert "AC(5)" in result
    assert "AC(20)" in result


def test_autocorr_custom_lags(random_walk):
    result = compute_autocorr(random_walk, lags=[1, 10])
    assert "AC(10)" in result
    assert "AC(20)" not in result


# ---------------------------------------------------------------------------
# Tests: compute_streaks
# ---------------------------------------------------------------------------
def test_streaks_returns_dict(trending_series):
    s = compute_streaks(trending_series)
    assert "max_consec_up" in s
    assert "max_consec_down" in s
    assert s["max_consec_up"] >= 0
    assert s["max_consec_down"] >= 0


def test_streaks_all_positive():
    """全正收益序列"""
    s = compute_streaks(pd.Series([0.01] * 10))
    assert s["max_consec_up"] == 10
    assert s["max_consec_down"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_trend_efficiency
# ---------------------------------------------------------------------------
def test_trend_efficiency_perfect():
    """完美单调上涨 → 效率接近 1"""
    s = pd.Series([0.001] * 100)
    r = compute_trend_efficiency(s)
    assert abs(r["trend_efficiency"]) > 0.9


def test_trend_efficiency_noisy(random_walk):
    """随机游走效率应接近 0"""
    r = compute_trend_efficiency(random_walk)
    assert -0.5 < r["trend_efficiency"] < 0.5


# ---------------------------------------------------------------------------
# Tests: compute_trend_stability
# ---------------------------------------------------------------------------
def test_trend_stability_range(trending_series):
    v = compute_trend_stability(trending_series)
    assert 0 <= v <= 1


# ---------------------------------------------------------------------------
# Tests: compute_ma_backtest
# ---------------------------------------------------------------------------
def test_ma_backtest_returns_dict(trending_series):
    r = compute_ma_backtest(trending_series)
    assert "ann_return" in r
    assert "sharpe" in r
    assert "max_drawdown" in r
    assert "win_rate" in r
    assert "num_trades" in r


def test_ma_backtest_trending_profitable(trending_series):
    """趋势序列上均线策略应盈利"""
    r = compute_ma_backtest(trending_series)
    assert r["ann_return"] > 0, f"Expected positive return, got {r['ann_return']}"
    assert r["sharpe"] > 0, f"Expected positive sharpe, got {r['sharpe']}"


# ---------------------------------------------------------------------------
# Tests: MomentumOperator (OOP wrapper)
# ---------------------------------------------------------------------------
def test_momentum_operator(trending_series):
    op = MomentumOperator()
    result = op.fit_transform(trending_series)
    assert isinstance(result, dict)
    assert "snr" in result
    assert "hurst" in result
