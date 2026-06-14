"""
测试 — 风险控制算子 (risk_control)

覆盖三种核心场景：
- 强趋势 + 低估值 → multiplier 应接近 1
- 强趋势 + 高估值 → multiplier 应受 penalize
- 均线死叉 + 高估值 → multiplier 应接近 0
"""

import numpy as np
import pandas as pd
import pytest

from operators.risk_control import (
    tsmom_signal,
    tsmom_strength,
    tsmom_ma_signal,
    tsmom_signal_binary,
    valuation_percentile,
    valuation_zscore,
    risk_multiplier,
    risk_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def strong_uptrend():
    """强上涨趋势 — 价格持续攀升"""
    np.random.seed(42)
    n = 500
    base = np.cumsum(np.full(n, 0.003))  # 每天 +0.3%
    noise = np.random.randn(n) * 0.005
    return pd.Series(base + noise + 1.0)


@pytest.fixture
def strong_downtrend():
    """强下跌趋势 — 价格持续下跌"""
    np.random.seed(42)
    n = 500
    base = np.cumsum(np.full(n, -0.003))
    noise = np.random.randn(n) * 0.005
    return pd.Series(base + 2.0)


@pytest.fixture
def sideways_market():
    """震荡行情 — 价格在区间内波动"""
    np.random.seed(42)
    n = 500
    base = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.1 + 1.0
    noise = np.random.randn(n) * 0.02
    return pd.Series(base + noise)


@pytest.fixture
def bubble_series():
    """先涨后跌 — 泡沫形成与破裂，用于测试估值百分位"""
    np.random.seed(42)
    n = 300
    # 前 200 天缓慢上涨
    slow_rise = np.linspace(1.0, 1.2, 200)
    # 后 100 天快速拉升 + 暴跌
    fast_rise = np.linspace(1.2, 1.8, 50)
    crash = np.linspace(1.8, 1.2, 50)
    series = np.concatenate([slow_rise, fast_rise, crash])
    noise = np.random.randn(n) * 0.01
    return pd.Series(series + noise)


# ===================================================================
# TSMOM 信号
# ===================================================================

def test_tsmom_signal_uptrend(strong_uptrend):
    """上涨趋势末尾应为多头信号"""
    s = tsmom_signal(strong_uptrend)
    tail = s.iloc[-10:]
    assert (tail == 1.0).all(), f"Expected all 1.0, got {tail.unique()}"


def test_tsmom_signal_downtrend(strong_downtrend):
    """下跌趋势末尾应为空头信号"""
    s = tsmom_signal(strong_downtrend)
    tail = s.iloc[-10:]
    assert (tail == -1.0).all(), f"Expected all -1.0, got {tail.unique()}"


def test_tsmom_signal_sideways(sideways_market):
    """震荡行情应包含正负交替"""
    s = tsmom_signal(sideways_market)
    uniq = s.iloc[-100:].unique()
    assert -1.0 in uniq or 0.0 in uniq, f"Expected negative/flat signal in sideways"


def test_tsmom_signal_output_series(strong_uptrend):
    s = tsmom_signal(strong_uptrend)
    assert isinstance(s, pd.Series)
    assert len(s) == len(strong_uptrend)


# ===================================================================
# TSMOM MA 信号
# ===================================================================

def test_tsmom_ma_uptrend(strong_uptrend):
    """上涨趋势应产生正信号"""
    s = tsmom_ma_signal(strong_uptrend)
    tail = s.iloc[-100:]
    assert tail.mean() > 0, f"Expected positive MA signal in uptrend, got {tail.mean()}"


def test_tsmom_ma_downtrend(strong_downtrend):
    """下跌趋势应产生负信号"""
    s = tsmom_ma_signal(strong_downtrend)
    tail = s.iloc[-100:]
    assert tail.mean() < 0, f"Expected negative MA signal in downtrend, got {tail.mean()}"


def test_tsmom_ma_clipped_range(strong_uptrend):
    """信号应在 [-1, 1] 范围内"""
    s = tsmom_ma_signal(strong_uptrend)
    assert s.min() >= -1.0 and s.max() <= 1.0


def test_tsmom_binary_uptrend(strong_uptrend):
    """上涨趋势 → 二元信号应为 1"""
    s = tsmom_signal_binary(strong_uptrend)
    tail = s.iloc[-50:]
    assert tail.mean() > 0.5, f"Expected mostly 1.0 in uptrend"


# ===================================================================
# 估值百分位
# ===================================================================

def test_valuation_percentile_output(strong_uptrend):
    v = valuation_percentile(strong_uptrend)
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert isinstance(v, pd.Series)


@pytest.mark.parametrize("method", ["minmax", "rank"])
def test_valuation_percentile_bubble(bubble_series, method):
    """泡沫顶部 → 百分位应接近 1；崩盘后 → 百分位下降"""
    v = valuation_percentile(bubble_series, method=method)
    # 第 250 个点（泡沫顶峰附近）
    peak_idx = 249 if len(bubble_series) > 250 else -1
    assert v.iloc[peak_idx] > 0.8, f"Expected high percentile at peak, got {v.iloc[peak_idx]}"


def test_valuation_percentile_rising(strong_uptrend):
    """持续上涨 → 末尾百分位应接近 1"""
    v = valuation_percentile(strong_uptrend)
    tail = v.iloc[-10:]
    assert tail.mean() > 0.9, f"Expected percentile near 1.0 at top of uptrend"


# ===================================================================
# 估值 Z-Score
# ===================================================================

def test_valuation_zscore_output(strong_uptrend):
    z = valuation_zscore(strong_uptrend)
    assert isinstance(z, pd.Series)
    assert not z.isna().all()


# ===================================================================
# 风险预算乘数
# ===================================================================

def test_multiplier_strong_uptrend_low_valuation():
    """场景1: 强趋势 + 低估值 → multiplier 应接近 1"""
    tsmom = pd.Series([1.0] * 10)
    val = pd.Series(np.linspace(0.1, 0.3, 10))  # 远低于阈值
    m = risk_multiplier(tsmom, val)
    tail_mean = m.iloc[-5:].mean()
    assert tail_mean > 0.8, f"Expected multiplier near 1.0, got {tail_mean}"


def test_multiplier_strong_uptrend_high_valuation():
    """场景2: 强趋势 + 高估值 → multiplier 受 penalty"""
    tsmom = pd.Series([1.0] * 10)
    val = pd.Series(np.linspace(0.85, 0.95, 10))  # 跨越阈值
    m = risk_multiplier(tsmom, val)
    # 估值高于阈值的部分应有 penalty
    assert m.iloc[-1] < m.iloc[0], "High valuation should reduce multiplier"


def test_multiplier_downtrend_high_valuation():
    """场景3: 死叉(负动量) + 高估值 → multiplier 应接近 0"""
    tsmom = pd.Series([-1.0] * 10)
    val = pd.Series([0.95] * 10)
    m = risk_multiplier(tsmom, val)
    assert m.iloc[-1] < 0.1, f"Expected near-zero multiplier, got {m.iloc[-1]}"


def test_multiplier_output_range():
    """所有场景输出应在 [0, 1]"""
    cases = [
        (pd.Series([1.0]), pd.Series([0.1])),
        (pd.Series([0.5]), pd.Series([0.5])),
        (pd.Series([0.0]), pd.Series([0.9])),
        (pd.Series([-1.0]), pd.Series([1.0])),
    ]
    for tsmom, val in cases:
        m = risk_multiplier(tsmom, val)
        assert 0.0 <= m.iloc[0] <= 1.0, f"Multiplier out of range: {m.iloc[0]}"


def test_multiplier_tsmom_signal_already_01():
    """tsmom 信号已在 [0, 1] 时不应被负数映射扭曲"""
    tsmom = pd.Series([0.3, 0.6, 0.9])
    val = pd.Series([0.3, 0.3, 0.3])
    m = risk_multiplier(tsmom, val)
    for i in range(len(tsmom)):
        assert m.iloc[i] >= tsmom.iloc[i] * 0.9, "Should preserve 0-1 tsmom signal"


# ===================================================================
# 风险状态
# ===================================================================

def test_risk_status_labels():
    m = pd.Series([0.1, 0.5, 0.9])
    labels = risk_status(m)
    assert labels.iloc[0] == "caution"
    assert labels.iloc[1] == "watch"
    assert labels.iloc[2] == "normal"


def test_risk_status_output_type():
    m = pd.Series([0.5])
    labels = risk_status(m)
    assert isinstance(labels, pd.Series)
    assert isinstance(labels.iloc[0], str)
