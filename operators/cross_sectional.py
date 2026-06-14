"""
横截面算子 — 资产间比较、排序与评分

包含：
- cross_sectional_rank: 多指标排名
- composite_score: 加权复合评分
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 横截面排名
# ---------------------------------------------------------------------------
def cross_sectional_rank(
    df: pd.DataFrame,
    rank_direction: Dict[str, bool],
) -> pd.DataFrame:
    """
    对 DataFrame 各列进行横截面排名。

    Parameters
    ----------
    df : pd.DataFrame
        包含各资产的指标值，每行一个资产，每列一个指标。
    rank_direction : Dict[str, bool]
        每个指标的排名方向：True = 越大越好，False = 越小越好。

    Returns
    -------
    pd.DataFrame
        排名结果，列名 = "{metric}_rank"。
    """
    rank_df = pd.DataFrame(index=df.index)
    for metric, higher_better in rank_direction.items():
        if metric not in df.columns:
            continue
        rank_df[f"{metric}_rank"] = (
            df[metric].rank(ascending=not higher_better)
        )
    return rank_df


# ---------------------------------------------------------------------------
# 复合评分
# ---------------------------------------------------------------------------
def composite_score(
    rank_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    加权求和排名得分为复合评分。

    Parameters
    ----------
    rank_df : pd.DataFrame
        包含 "{metric}_rank" 列的排名 DataFrame。
    weights : Dict[str, float]
        各指标权重，如 {"SNR": 0.15, "Hurst": 0.15}。
        权重无需归一化。

    Returns
    -------
    pd.DataFrame
        新增 "composite_score" 和 "overall_rank" 列。
    """
    result = rank_df.copy()
    result["composite_score"] = 0.0
    for metric, w in weights.items():
        col = f"{metric}_rank"
        if col in rank_df.columns:
            result["composite_score"] += w * rank_df[col]
    result["overall_rank"] = result["composite_score"].rank(ascending=True)
    return result


# ===================================================================
# 算子类封装
# ===================================================================

class CrossSectionalRankOperator:
    """横截面排名算子 — 对多资产多指标执行排名与复合评分"""

    def __init__(
        self,
        rank_direction: Optional[Dict[str, bool]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.rank_direction = rank_direction or {}
        self.weights = weights or {}

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行排名"""
        return cross_sectional_rank(df, self.rank_direction)

    def score(self, rank_df: pd.DataFrame) -> pd.DataFrame:
        """计算复合评分"""
        return composite_score(rank_df, self.weights)

    def rank_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """排名 + 评分链式调用"""
        return self.score(self.rank(df))

    def __repr__(self) -> str:
        return f"<CrossSectionalRankOperator (metrics={len(self.rank_direction)})>"
