"""
动量/趋势类算子 — Formulaic Operators

每个函数 = 一个原子公式，纯向量化，零显式循环。
输入 pd.Series → 输出 float / Dict[str, float] / Dict[str, int]。

参考标准：WorldQuant BRAIN Formulaic Operators 工程规范。
在大类资产配置场景下，
用月线构建的时序动量比用日线或周线构建的时序动量的整体效果要好，用时序动量识别趋势破位比用时序动量识别趋势突破的准确率要高。
从行为金融学的角度来解释，可能跟非理性投资者的“处置效应”和“锚定效应”有关系。所谓处置效应，是指投资者倾向于卖出浮盈的资产而继续持有浮亏的资产，导致浮亏筹码的换手不充分、出清过程漫长；所谓锚定效应，是指当浮亏的资产重新涨回成本价时，部分投资者会果断卖出收回本金，导致上涨过程阻力重重。这些可能是大类资产中长期空头趋势更容易得到延续的原因。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from operators.base import nanmask

BDAYS_PER_YEAR = 252


def ts_multi_dimensional_momentum_operator(
    daily_prices_df: pd.DataFrame,
    ma_days: list = [20, 60, 120],
    fast_month: int = 5,
    slow_month: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0
) -> pd.DataFrame:
    """
    WorldQuant BRAIN 风格：多维时序绝对动量融合算子 (TS-Multi-Mom)
    
    输入：
        daily_prices_df: 日度收盘价矩阵 (DataFrame)，index为DatetimeIndex，columns为资产代码
    输出：
        risk_multipliers: 与输入维度一致的月频风险预算乘数矩阵 (DataFrame)，值域 [0, 1]
    """
    # 确保没有空值干扰计算
    df_daily = daily_prices_df.ffill().bfill()
    
    # =========================================================================
    # 1. 日线级别算子：均线相对位置 (MA Distance) & MACD
    # =========================================================================
    # 计算日线 MA 偏离度：(Price / MA) - 1
    ma_scores = pd.DataFrame(0.0, index=df_daily.index, columns=df_daily.columns)
    for ma in ma_days:
        ma_series = df_daily.rolling(window=ma, min_periods=ma).mean()
        # 价格在均线上方贡献正分，下方贡献负分
        ma_scores += (df_daily / ma_series - 1.0)
    # 归一化均线得分 (简单的符号化或缩放，这里采用符号映射：多头市场得 1，空头市场得 0)
    ma_signal = np.where(ma_scores > 0, 1.0, 0.0)
    ma_signal_df = pd.DataFrame(ma_signal, index=df_daily.index, columns=df_daily.columns)

    # 日线 MACD 计算
    exp_fast = df_daily.ewm(span=macd_fast, adjust=False).mean()
    exp_slow = df_daily.ewm(span=macd_slow, adjust=False).mean()
    macd_dif = exp_fast - exp_slow
    macd_dea = macd_dif.ewm(span=macd_signal, adjust=False).mean()
    macd_hist = macd_dif - macd_dea
    # MACD 处于绿柱且绿柱扩大时（柱状图 < 0 且今日值比昨日值更小），提示趋势破位得 0.0；其余情况为 1.0
    macd_prev = macd_hist.shift(1)
    macd_signal = np.where((macd_hist < 0) & (macd_hist < macd_prev), 0.0, 1.0)
    macd_signal_df = pd.DataFrame(macd_signal, index=df_daily.index, columns=df_daily.columns)

    # =========================================================================
    # 2. 月线级别算子：双均线交叉 (Monthly MA Cross) & 月线布林带 (Monthly Bollinger Bands)
    # =========================================================================
    df_monthly = df_daily.resample('ME').last()
    
    # 2.1 月线双均线交叉
    m_ma_fast = df_monthly.rolling(window=fast_month, min_periods=fast_month).mean()
    m_ma_slow = df_monthly.rolling(window=slow_month, min_periods=slow_month).mean()
    # 金叉（快线 > 慢线）得 1.0，死叉得 0.0，初始缺失值设为 NaN 以便后续兜底
    m_ma_signal = np.where(m_ma_fast.isna() | m_ma_slow.isna(), np.nan, np.where(m_ma_fast > m_ma_slow, 1.0, 0.0))
    m_ma_signal_df = pd.DataFrame(m_ma_signal, index=df_monthly.index, columns=df_monthly.columns)

    # 2.2 月线布林带
    m_bb_mid = df_monthly.rolling(window=bb_window, min_periods=bb_window).mean()
    m_bb_std_dev = df_monthly.rolling(window=bb_window, min_periods=bb_window).std()
    m_bb_upper = m_bb_mid + bb_std * m_bb_std_dev
    m_bb_lower = m_bb_mid - bb_std * m_bb_std_dev
    # 从上到下穿越中轨时提示趋势破位 (得 0.0)；但突破下轨时是反转信号，不能再看空 (得 1.0)；其余（中轨上方）得 1.0
    m_bb_signal = np.where(
        m_bb_mid.isna() | m_bb_lower.isna(),
        np.nan,
        np.where((df_monthly >= m_bb_mid) | (df_monthly <= m_bb_lower), 1.0, 0.0)
    )
    m_bb_signal_df = pd.DataFrame(m_bb_signal, index=df_monthly.index, columns=df_monthly.columns)

    # =========================================================================
    # 3. 多维动量合成与时间轴对齐 (Monthly Resample)
    # =========================================================================
    # 日线指标综合 (均值融合，包含均线和MACD两个维度)
    daily_combined = (ma_signal_df + macd_signal_df) / 2.0
    
    # 将日线综合指标降采样至月频（取月底那天的状态）
    monthly_daily_combined = daily_combined.resample('ME').last()
    
    # 月线指标综合 (均值融合，包含双均线交叉和布林带两个维度)
    monthly_combined = (m_ma_signal_df + m_bb_signal_df) / 2.0
    
    # 融合月线指标：日线指标占 60% 权重，月线长周期指标占 40% 权重
    final_momentum_score = (monthly_daily_combined * 0.6) + (monthly_combined * 0.4)
    
    # 兜底处理：由于滚动窗口导致的初始缺失值，默认不扣减风险预算（填 1.0）
    final_momentum_score = final_momentum_score.fillna(1.0)
    
    # 严格保证输出在 [0.0, 1.0] 之间，且保持严格的 DateTimeIndex
    return final_momentum_score

# ===================================================================
# 滞后自相关 | Lag-k Autocorrelation
# ===================================================================

def acf(series: pd.Series, lag: int = 1) -> float:
    r"""
    滞后 k 阶自相关系数。

    Formula:
        ρ(k) = Cov(r_t, r_{t-k}) / σ(r_t) · σ(r_{t-k})

    Parameters
    ----------
    series : pd.Series
        日收益率序列。
    lag : int
        滞后阶数。

    Returns
    -------
    float
    """
    return series.autocorr(lag=lag)


def acf_multi(series: pd.Series, lags: Optional[List[int]] = None) -> Dict[str, float]:
    """
    多阶滞后自相关系数（向量化聚合）。

    Parameters
    ----------
    series : pd.Series
    lags : list of int, default [1, 5, 20]

    Returns
    -------
    {"AC(1)": ..., "AC(5)": ..., "AC(20)": ...}
    """
    if lags is None:
        lags = [1, 5, 20]
    return {f"AC({lag})": series.autocorr(lag=lag) for lag in lags}


# ===================================================================
# 信噪比 | Signal-to-Noise Ratio
# ===================================================================

@nanmask(min_periods=20)
def snr(series: pd.Series) -> float:
    r"""
    信噪比 SNR。

    Formula:
        SNR = (252 · μ) / (√252 · σ)

    越高表示趋势收益越突出。
    """
    s = series.dropna()
    ann_mean = s.mean() * BDAYS_PER_YEAR
    ann_vol = s.std() * np.sqrt(BDAYS_PER_YEAR)
    return ann_mean / ann_vol if ann_vol > 0 else np.nan


# ===================================================================
# Hurst 指数 | Hurst Exponent (R/S 分析法，纯向量化)
# ===================================================================

@nanmask(min_periods=100)
def hurst_exponent(series: pd.Series, min_lag: int = 4) -> float:
    r"""
    Hurst 指数 — 纯向量化 R/S 分析，零显式循环。

    Formula:
        H = lim_{n→∞} log(R/S) / log(n)

    解读:
        H > 0.5 → 趋势性    H < 0.5 → 均值回归
        H ≈ 0.5 → 随机游走

    实现：所有分块的 R/S 在一个 numpy 操作中并行计算。
    """
    data = np.asarray(series.dropna().values, dtype=np.float64)
    n = len(data)
    max_lag = n // 4

    # 生成所有分块大小 (对数均匀分布)
    lags = np.unique(
        np.geomspace(min_lag, max_lag, 50).astype(np.int64)
    )
    lags = lags[lags >= 4]

    if len(lags) < 6:
        return np.nan

    # 对所有 lag 值执行 R/S 计算：每行 = 一种 lag 的结果
    # 使用 numpy 向量化，避免 Python for-loop
    rs_means = _rs_vectorized(data, lags)
    valid = ~np.isnan(rs_means)

    if valid.sum() < 4:
        return np.nan

    coeffs = np.polyfit(np.log(lags[valid]), np.log(rs_means[valid]), 1)
    return float(coeffs[0])


def _rs_vectorized(data: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """
    对所有 lag 值并行计算 R/S 均值。

    每个 lag 被拆分为 floor(n / lag) 个不重叠分段，
    所有分段运算通过纯 numpy 广播实现。
    """
    n = len(data)
    results = np.full(len(lags), np.nan)

    for i, lag in enumerate(lags):
        n_blocks = n // lag
        if n_blocks < 2:
            continue

        # 重塑为 (n_blocks, lag) — 一次视图操作，无数据拷贝
        blocks = data[:n_blocks * lag].reshape(n_blocks, lag)

        # 全向量化 R/S 计算
        # step 1: 每块均值
        means = blocks.mean(axis=1, keepdims=True)

        # step 2: 偏差累积和
        deviations = blocks - means
        cumsums = np.cumsum(deviations, axis=1)

        # step 3: R = max - min, S = std
        R = cumsums.max(axis=1) - cumsums.min(axis=1)
        S = np.std(blocks, axis=1, ddof=1)

        # step 4: 均值
        valid_mask = (S > 0) & (R > 0)
        if valid_mask.any():
            results[i] = np.mean(R[valid_mask] / S[valid_mask])

    return results


# ===================================================================
# 连涨连跌极值 | Max Consecutive Streaks
# ===================================================================

def streaks(series: pd.Series) -> Dict[str, int]:
    """
    最长连涨 / 连跌天数 — 全向量化实现。

    原理：利用 diff 检测方向变化点，用 groupby-cumcount 计算每段连续长度。

    Returns
    -------
    {"max_consec_up": int, "max_consec_down": int}
    """
    s = series.dropna().values
    if len(s) == 0:
        return {"max_consec_up": 0, "max_consec_down": 0}

    # 方向标签: +1 ↗, -1 ↘, 0 不变
    direction = np.sign(s)

    # 变化点: 相邻方向不同
    changes = np.diff(direction, prepend=direction[0])
    change_points = np.cumsum(changes != 0)

    # 利用 pandas groupby 代替 for-loop
    grouped = pd.Series(direction).groupby(change_points)

    max_up = (
        grouped.apply(lambda g: g.eq(1).sum()).max()
        if direction.max() >= 0 else 0
    )
    max_down = (
        grouped.apply(lambda g: g.eq(-1).sum()).max()
        if direction.min() <= 0 else 0
    )

    return {
        "max_consec_up": int(max_up) if not np.isnan(max_up) else 0,
        "max_consec_down": int(max_down) if not np.isnan(max_down) else 0,
    }


# ===================================================================
# 趋势效率 | Trend Efficiency
# ===================================================================

@nanmask(min_periods=10)
def trend_efficiency(series: pd.Series) -> float:
    r"""
    趋势效率 η。

    Formula:
        η = [∏(1 + r_t) - 1] / Σ|r_t|

    接近 ±1 → 方向高度一致（适合趋势跟踪）。
    接近 0   → 价格来回震荡（不适合趋势跟踪）。
    """
    s = series.dropna()
    cum_ret = (1 + s).prod() - 1
    abs_sum = s.abs().sum()
    return cum_ret / abs_sum if abs_sum > 0 else 0.0


@nanmask(min_periods=10)
def trend_strength(series: pd.Series) -> float:
    r"""
    趋势强度 — 累计绝对收益。

    Formula:
        S = |∏(1 + r_t) - 1|

    衡量趋势的"幅度"而非"方向一致性"。
    """
    cum_ret = (1 + series).prod() - 1
    return float(abs(cum_ret))


# ===================================================================
# 趋势稳定性 | Trend Stability
# ===================================================================

@nanmask(min_periods=60)
def trend_stability(
    series: pd.Series,
    fast: int = 20,
    slow: int = 60,
) -> float:
    r"""
    趋势稳定性 — 快慢均线斜率同方向比例。

    Formula:
        Stability = (1/N) · Σ I{sign(ΔSMA_f) = sign(ΔSMA_s)}

    取值 (0, 1)，越接近 1 → 方向一致性越高。
    """
    price = (1 + series.dropna()).cumprod()
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()

    slope_fast = sma_fast.diff()
    slope_slow = sma_slow.diff()

    valid = ~(slope_fast.isna() | slope_slow.isna())
    if valid.sum() == 0:
        return np.nan

    consistent = np.sign(slope_fast[valid]) == np.sign(slope_slow[valid])
    return float(consistent.mean())


# ===================================================================
# 双均线交叉策略 | MA Crossover Backtest
# ===================================================================

@nanmask(min_periods=60)
def ma_crossover_metrics(
    series: pd.Series,
    fast: int = 20,
    slow: int = 60,
) -> Dict[str, float]:
    r"""
    双均线趋势跟踪策略的绩效指标。

    Rules:
        signal_{t} = sign(SMA_fast_{t-1} - SMA_slow_{t-1})

    Returns
    -------
    年化收益 / 波动 / 夏普 / 最大回撤 / 胜率 / 交易次数 / Calmar
    """
    s = series.dropna()
    price = (1 + s).cumprod()
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()

    # 前一日均线决定今日持仓
    signal = pd.Series(0.0, index=s.index)
    cond_long = sma_fast.shift(1) > sma_slow.shift(1)
    cond_short = sma_fast.shift(1) < sma_slow.shift(1)
    signal[cond_long] = 1.0
    signal[cond_short] = -1.0

    strat_ret = signal * s
    in_market = signal != 0
    sr = strat_ret[in_market]

    if len(sr) == 0:
        return _nan_metrics()

    nav = (1 + sr).cumprod()
    n_years = len(sr) / BDAYS_PER_YEAR
    ann_ret = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
    ann_vol = float(sr.std() * np.sqrt(BDAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    running_max = nav.expanding().max()
    dd = nav / running_max - 1
    max_dd = float(dd.min())

    win_rate = float((sr > 0).sum() / len(sr))
    num_trades = int((signal.diff() != 0).sum() // 2)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "calmar": calmar,
    }


def _nan_metrics() -> Dict[str, float]:
    """空交易时的占位返回值"""
    return {k: np.nan for k in
            ["ann_return", "ann_vol", "sharpe", "max_drawdown",
             "win_rate", "num_trades", "calmar"]}
