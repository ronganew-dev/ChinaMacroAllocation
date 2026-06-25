"""
中国宏观数据统一清洗与预处理算子 — China Macro Data Cleanser Operators

每个函数设计为一个原子公式，采用纯向量化或高能效的 pandas/numpy 表达，支持 pandas.Series 输入，输出与输入形状及索引一致的 pandas.Series。

"""

from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ── 1. 统一口径处理 (Caliber Conversion) ────────────────────────────────────────

def ts_diff(series: pd.Series, period: int = 12) -> pd.Series:
    """
    差分算子 (Difference)。
    
    Formula:
        Diff_t = Y_t - Y_{t-period}
    """
    return series.diff(periods=period)


def ts_pct_change(series: pd.Series, period: int = 12) -> pd.Series:
    """
    变化率算子 (Percentage Change)。
    
    Formula:
        PctChange_t = (Y_t / Y_{t-period}) - 1
    """
    return series.pct_change(periods=period)


def ts_mom_to_yoy(series: pd.Series, is_percentage: bool = False) -> pd.Series:
    """
    环比增长率转换为同比增长率。
    
    Formula:
        YoY_t = prod_{i=0}^{11} (1 + MoM_{t-i}) - 1
        
    参数:
    ----------
    series : pd.Series
        月频环比数据系列（MoM）。
    is_percentage : bool, 默认 False
        输入数据是否已乘100（即以百分数表示，如1.5表示1.5%）。如果是，输出也将保持百分数格式。
    """
    factor = 100.0 if is_percentage else 1.0
    s_decimal = series / factor
    yoy = (1.0 + s_decimal).rolling(window=12).apply(np.prod, raw=True) - 1.0
    return yoy * factor


def ts_yoy_to_mom_decompound(series: pd.Series, is_percentage: bool = False) -> pd.Series:
    """
    同比增长率去复利转换为环比增长率（Decompounded Approximation）。
    在无法获取底层高频绝对值时，基于同比增速复利平均分配的近似处理。
    
    Formula:
        MoM_t = (1 + YoY_t)^(1/12) - 1
        
    参数:
    ----------
    series : pd.Series
        月频同比数据系列（YoY）。
    is_percentage : bool, 默认 False
        输入数据是否已乘100（即以百分数表示）。
    """
    factor = 100.0 if is_percentage else 1.0
    s_decimal = series / factor
    # 避免对负数进行1/12次方（在极端经济下行或通缩时可能出现YoY < -1，通常不应发生）
    s_decimal_clipped = s_decimal.clip(lower=-0.9999)
    mom = (1.0 + s_decimal_clipped) ** (1.0 / 12.0) - 1.0
    return mom * factor


# ── 2. 缺失值处理 (Missing Value Imputation) ──────────────────────────────────

def ts_impute_ffill(series: pd.Series) -> pd.Series:
    """
    前向填充与后向填充（最基础的缺失值填充）。
    无未来信息（ffill部分），bfill部分作为头部缺失的妥协。
    """
    return series.ffill().bfill()


def ts_impute_linear(series: pd.Series) -> pd.Series:
    """
    线性插值填充（注意：线性插值会引入未来信息，仅适用于历史分析或低频宏观因子的插值）。
    """
    return series.interpolate(method='linear').ffill().bfill()


def ts_impute_rolling(series: pd.Series, window: int = 12, method: str = 'mean') -> pd.Series:
    """
    基于滚动历史统计值的缺失值填充（无未来信息，适合实盘与回测）。
    
    参数:
    ----------
    series : pd.Series
        待填充序列。
    window : int, 默认 12
        滚动历史窗口大小。
    method : str, 默认 'mean'
        填充方法，可选 'mean' (均值) 或 'median' (中位数)。
    """
    s = series.copy()
    if method == 'mean':
        roll_stat = s.shift(1).rolling(window=window, min_periods=1).mean()
    elif method == 'median':
        roll_stat = s.shift(1).rolling(window=window, min_periods=1).median()
    else:
        raise ValueError("method must be 'mean' or 'median'")
    
    return s.fillna(roll_stat).ffill().bfill()


def ts_impute_seasonal(series: pd.Series, max_years: int = 3) -> pd.Series:
    """
    季节性匹配填充。对于 t 月的缺失值，使用过去若干年相同月份（t-12, t-24, t-36...）的均值进行填充。
    对于强季节性宏观数据（如出口、工业增加值）最合理的填充方式。
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            raise ValueError("Series index must be DatetimeIndex or convertible to DatetimeIndex")
            
    months = s.index.month
    years = s.index.year
    nan_indices = s[s.isna()].index
    
    for idx in nan_indices:
        m = idx.month
        y = idx.year
        past_vals = []
        for dy in range(1, max_years + 1):
            past_y = y - dy
            past_idx = s.index[(s.index.month == m) & (s.index.year == past_y)]
            if not past_idx.empty:
                val = s.loc[past_idx[-1]]
                if not pd.isna(val):
                    past_vals.append(val)
        if past_vals:
            s.loc[idx] = np.mean(past_vals)
            
    # 对头部由于没有历史月份而残留的 NaN 进行兜底填充
    return s.ffill().bfill()


# ── 3. 春节效应消除 (Spring Festival Adjustment) ─────────────────────────────

# 中国春节公历日期对照表 (2000 - 2030)
CNY_DATES = {
    2000: "2000-02-05", 2001: "2001-01-24", 2002: "2002-02-12", 2003: "2003-02-01",
    2004: "2004-01-22", 2005: "2005-02-09", 2006: "2006-01-29", 2007: "2007-02-18",
    2008: "2008-02-07", 2009: "2009-01-26", 2010: "2010-02-14", 2011: "2011-02-03",
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17", 2027: "2027-02-06",
    2028: "2028-01-26", 2029: "2029-02-13", 2030: "2030-02-03"
}

def ts_cny_combine_jan_feb(series: pd.Series, method: str = 'mean') -> pd.Series:
    """
    春节效应消除：合并1月与2月数据。
    中国官方（国家统计局）通常会对1-2月指标合并发布或不单独公布1月。
    该算子将同一年度的1月和2月数值替换为二者的平均值或总和，从根本上抹平春节错位带来的剧烈波动。
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
        
    years = s.index.year.unique()
    for y in years:
        mask_y = s.index.year == y
        mask_jf = mask_y & s.index.month.isin([1, 2])
        jf_data = s[mask_jf]
        if not jf_data.empty:
            if method == 'mean':
                val = jf_data.mean()
            elif method == 'sum':
                val = jf_data.sum()
            else:
                raise ValueError("method must be 'mean' or 'sum'")
            s.loc[mask_jf] = val
    return s


def ts_cny_regression_adjust(series: pd.Series) -> pd.Series:
    """
    春节效应回归调整。
    以春节法定长假（除夕至正月初五共7天）在1月和2月的分布天数作为哑变量，通过回归剥离春节长假干扰，提取残差。
    
    Formula:
        Y_t = alpha + beta * CNY_Days_t + epsilon_t
        Y_adjusted_t = Y_t - beta * CNY_Days_t
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
        
    cny_dummy = pd.Series(0.0, index=s.index)
    
    for y in s.index.year.unique():
        if y in CNY_DATES:
            cny_dt = pd.to_datetime(CNY_DATES[y])
            # 春节长假一般从除夕(CNY-1)到正月初五(CNY+5)
            h_start = cny_dt - pd.Timedelta(days=1)
            h_days = pd.date_range(start=h_start, periods=7)
            
            jan_days = sum(1 for d in h_days if d.month == 1)
            feb_days = sum(1 for d in h_days if d.month == 2)
            
            mask_jan = (s.index.year == y) & (s.index.month == 1)
            mask_feb = (s.index.year == y) & (s.index.month == 2)
            
            cny_dummy.loc[mask_jan] = jan_days
            cny_dummy.loc[mask_feb] = feb_days
            
    valid = s.notna()
    if valid.sum() < 3:
        return s
        
    X = sm.add_constant(cny_dummy[valid])
    model = sm.OLS(s[valid], X).fit()
    
    beta = model.params.iloc[1] if len(model.params) > 1 else 0.0
    adjusted = s - beta * cny_dummy
    return adjusted


# ── 4. 季节性调整 (Seasonal Adjustment) ────────────────────────────────────────

def ts_seasonal_adjust_classical(series: pd.Series, model: str = 'additive') -> pd.Series:
    """
    经典双向季节性调整（使用 statsmodels 的 seasonal_decompose）。
    警告：这属于双向滤波，存在未来信息泄漏，仅用于历史归因或学术研究，不可直接用于实时回测/实盘。
    """
    s = series.copy()
    nan_mask = s.isna()
    s_filled = s.ffill().bfill()
    
    if len(s_filled) < 24:
        # 数据过短无法进行经典季节分解 (至少需要2个周期)
        return series
        
    res = seasonal_decompose(s_filled, model=model, period=12, extrapolate_trend='freq')
    
    if model == 'additive':
        sadj = s_filled - res.seasonal
    elif model == 'multiplicative':
        seasonal_safe = np.where(res.seasonal == 0, 1.0, res.seasonal)
        sadj = s_filled / seasonal_safe
    else:
        raise ValueError("model must be 'additive' or 'multiplicative'")
        
    sadj[nan_mask] = np.nan
    return sadj


def ts_seasonal_adjust_causal(series: pd.Series, window: int = 12) -> pd.Series:
    """
    单向/因果季节性调整（无未来信息透视，适合实盘与回测）。
    
    Formula:
        SADJ_t = Y_t - Mean(Y_{t-1}, ..., Y_{t-window})  [加法模型]
    此处取 window=12，即减去过去 12 个月的滚动平均，可有效剥离月度季节均值漂移。
    """
    return series - series.rolling(window=window, min_periods=1).mean()


# ── 5. HP 滤波 (Hodrick-Prescott Filtering) ────────────────────────────────────

def ts_hp_filter_one_sided(series: pd.Series, lamb: float = 14400, return_type: str = 'trend') -> pd.Series:
    """
    单向/因果 HP 滤波 (One-sided HP Filter)。
    对 t 时刻的趋势/循环项估计，仅使用 [0, t] 期间的数据，取双向HP滤波在 t 时刻的截面末端值。
    无未来信息泄漏，非常适合实盘构建趋势跟踪/动量指标。
    
    参数:
    ----------
    series : pd.Series
        输入时间序列。
    lamb : float, 默认 14400 (月频数据的经典平滑参数)
        平滑系数。
    return_type : str, 默认 'trend'
        返回 'trend' (趋势项) 或者是 'cycle' (循环项)。
    """
    s = series.copy()
    nan_mask = s.isna()
    s_valid = s.dropna()
    
    if len(s_valid) < 3:
        return pd.Series(np.nan, index=series.index)
        
    s_arr = s_valid.values
    n = len(s_arr)
    trend = np.zeros(n)
    cycle = np.zeros(n)
    
    # 递归/前向截面计算
    for i in range(n):
        if i < 3:
            trend[i] = s_arr[i]
            cycle[i] = 0.0
        else:
            sub_x = s_arr[:i+1]
            sub_c, sub_t = sm.tsa.filters.hpfilter(sub_x, lamb)
            trend[i] = sub_t[-1]
            cycle[i] = sub_c[-1]
            
    res_vals = trend if return_type == 'trend' else cycle
    res_series = pd.Series(res_vals, index=s_valid.index)
    return res_series.reindex(series.index)


def ts_hp_filter_two_sided(series: pd.Series, lamb: float = 14400, return_type: str = 'trend') -> pd.Series:
    """
    双向经典 HP 滤波 (Two-sided HP Filter)。
    利用全样本信息对每一时刻进行平滑。存在未来信息泄漏，仅适用于历史 regime 划分与因子归因。
    """
    s = series.copy()
    nan_mask = s.isna()
    s_valid = s.dropna()
    
    if len(s_valid) < 3:
        return pd.Series(np.nan, index=series.index)
        
    cycle, trend = sm.tsa.filters.hpfilter(s_valid, lamb)
    res_series = trend if return_type == 'trend' else cycle
    return res_series.reindex(series.index)


# ── 6. 辅助及归一化 (Scaling & Smoothing) ───────────────────────────────────────

def ts_min_max_scale_global(series: pd.Series) -> pd.Series:
    """
    全局 0-1 极差标准化。
    注意：使用了全样本的 Max/Min，属于双向处理，有未来数据泄露。
    
    Formula:
        Scaled_t = (Y_t - Min(Y)) / (Max(Y) - Min(Y))
    """
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def ts_min_max_scale_rolling(series: pd.Series, window: int = 12) -> pd.Series:
    """
    单向/滚动 0-1 极差标准化 (Causal Rolling Min-Max Scaling)。
    利用过去 window 期的最大最小值进行标准化，无未来信息泄露。
    
    Formula:
        Scaled_t = (Y_t - Min(Y_{t-w+1..t})) / (Max(Y_{t-w+1..t}) - Min(Y_{t-w+1..t}))
    """
    s = series.copy()
    roll_min = s.rolling(window=window, min_periods=2).min()
    roll_max = s.rolling(window=window, min_periods=2).max()
    
    diff = roll_max - roll_min
    # 避免除以 0
    scaled = np.where(diff > 0, (s - roll_min) / diff, 0.5)
    return pd.Series(scaled, index=s.index)


def ts_wma(series: pd.Series, window: int = 6) -> pd.Series:
    """
    时间加权移动平均 (Time-Weighted Moving Average / Linear Decay MA)。
    越靠近当前时刻，权重及权重索引越大。
    """
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
        raw=True
    )
