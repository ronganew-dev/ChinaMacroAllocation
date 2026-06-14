"""
横截面算子 — Formulaic Operators

rank(axis=1, pct=True, method='first')
保证百分比排名（Percentile Rank）的均匀分布： 当设置 pct=True 时，如果使用 method='first'，无论是否有相同的值，最终返回的百分比排名一定会均匀分布在 $[1/N, 1.0]$ 区间内。而默认的 average 会让并列的值挤在同一个分位数上。

维持多空组合权重的对称性与稳定性： 截面多空信号需要将排名中心化（减去 0.5）以确保多头和空头权重对称。如果并列的值很多且使用 average，可能会导致截面去均值（Demean）后，多空双方的资产个数和权重严重不对称。使用 first 能确保在任何时候，排名的数字集合都是稳定的 ${1, 2, \dots, N}$，从而使多空组合的截面权重分配非常均衡和稳定。

防止零收益率（停牌）导致仓位失衡： 在实际交易中，经常会遇到多个资产在某天收益率为 0（例如同时停牌或无交易）。使用 first 可以非常安全地把这些 0 均匀地分摊到多头和空头侧，而不会导致所有停牌资产被分配到完全相同的权重，从而规避了潜在的系统奇异值
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ===================================================================
# 横截面排名 | Cross-Sectional Ranking
# ===================================================================

def cross_sectional_rank(
    df: pd.DataFrame,
    rank_direction: Dict[str, bool],
) -> pd.DataFrame:
    r"""
    多指标横截面排名 — 全程向量化。

    对 DataFrame 的每列（每个指标）在所有资产间独立排名。
    排名方向可配置：True = 越大越好，False = 越小越好。

    Parameters
    ----------
    df : pd.DataFrame
        行 = 资产，列 = 指标。
    rank_direction : Dict[str, bool]
        {指标名: 是否越大越好}。

    Returns
    -------
    pd.DataFrame, 列名 = "{metric}_rank"。
    """
    cols = [m for m in rank_direction if m in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    # 一次向量化 rank() 全量调用，然后逐个调整方向
    # 组合为单个 DataFrame，避免逐列 .rank() 调用
    data = pd.DataFrame(index=df.index)
    for metric in cols:
        asc = not rank_direction[metric]
        data[f"{metric}_rank"] = df[metric].rank(ascending=asc)

    return data


def cross_sectional_zscore(
    df: pd.DataFrame,
    cap: Optional[float] = 5.0,
) -> pd.DataFrame:
    r"""
    横截面 Z-Score 标准化。

    Formula:
        z_i = (x_i - μ_x) / σ_x

    其中 μ_x 和 σ_x 为截面均值与标准差。
    支持 outlier capping。

    Parameters
    ----------
    df : pd.DataFrame
        行 = 资产，列 = 指标。
    cap : float or None
        Z-Score 截断上限，None 为不截断。

    Returns
    -------
    pd.DataFrame, 同 shape，中心化 + 标准化。
    """
    mean = df.mean(axis=0)
    std = df.std(axis=0, ddof=1)
    z = (df - mean) / std.replace(0, np.nan)

    if cap is not None:
        z = z.clip(-cap, cap)

    return z


# ===================================================================
# 复合评分 | Composite Score
# ===================================================================

def composite_score(
    rank_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    r"""
    加权复合评分 — 全程矩阵向量化。

    Formula:
        score_i = Σ_j w_j · rank_{i,j}

        overall_rank = rank(score_i)  (1 = 最高分)

    Parameters
    ----------
    rank_df : pd.DataFrame
        含 "{metric}_rank" 列的排名结果。
    weights : Dict[str, float]
        {指标名: 权重}，无需归一化。

    Returns
    -------
    pd.DataFrame, 新增 composite_score 和 overall_rank 列。
    """
    result = rank_df.copy()

    # 解析可匹配的指标列与对应权重（结构配置，非数据循环）
    matched = [
        (m, f"{m}_rank", w)
        for m, w in weights.items()
        if f"{m}_rank" in rank_df.columns
    ]

    if not matched:
        result["composite_score"] = np.nan
        result["overall_rank"] = np.nan
        return result

    cols = [col for _, col, _ in matched]
    w_arr = np.array([w for _, _, w in matched])

    # 单次矩阵乘法 Σ w_j · rank_{i,j} — 全向量化
    score = rank_df[cols].values @ w_arr
    result["composite_score"] = score
    result["overall_rank"] = pd.Series(score, index=rank_df.index).rank(ascending=True).astype(int)

    return result


# ===================================================================
# 行业中性化 | Sector Neutralization (Cross-Sectional)
# ===================================================================

def sector_neutralize(
    values: pd.Series,
    sector_labels: pd.Series,
    method: str = "demean",
) -> pd.Series:
    r"""
    行业中性化。

    Parameters
    ----------
    values : pd.Series
        各资产的原始值，index = 资产名。
    sector_labels : pd.Series
        行业标签，index = 资产名。
    method : str
        "demean" → 减去行业均值
        "zscore" → 行业内标准化

    Returns
    -------
    pd.Series, 中性化后的值。
    """
    if method == "demean":
        sector_mean = values.groupby(sector_labels).transform("mean")
        return values - sector_mean
    elif method == "zscore":
        sector_mean = values.groupby(sector_labels).transform("mean")
        sector_std = values.groupby(sector_labels).transform("std").replace(0, np.nan)
        return (values - sector_mean) / sector_std
    else:
        raise ValueError(f"Unknown method: {method}")
