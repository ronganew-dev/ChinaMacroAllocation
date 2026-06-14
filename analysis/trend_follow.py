"""
趋势跟踪适合度定量分析

对多资产进行趋势跟踪策略的适合度评估，输出结构化指标与综合排名。

使用 operators 中的通用算子完成所有计算。
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from operators.momentum import (
    compute_autocorr,
    compute_hurst,
    compute_ma_backtest,
    compute_snr,
    compute_streaks,
    compute_trend_efficiency,
    compute_trend_stability,
)
from operators.cross_sectional import composite_score, cross_sectional_rank

BDAYS_PER_YEAR = 252

# ──────────────────────────────────────────────
# 资产配置 (可被外部覆盖)
# ──────────────────────────────────────────────
ASSET_COL_INDEX: Dict[str, int] = {
    "沪深300": 7,
    "标普500": 17,
    "量化中性": 10,
    "黄金": 4,
    "中证商品指数": 15,
    "国债30年": 6,
}

ASSET_NAMES_EN: Dict[str, str] = {
    "沪深300": "CSI 300",
    "标普500": "S&P 500",
    "量化中性": "Quant Neutral",
    "黄金": "Gold",
    "中证商品指数": "CSI Commodity",
    "国债30年": "30Y Treasury",
}

# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_returns(input_path: str, sheet: str = "return") -> Dict[str, pd.Series]:
    """
    从 Excel 读取各资产日收益率序列。

    Parameters
    ----------
    input_path : str
        Excel 文件路径。
    sheet : str
        Sheet 名称，默认 "return"。

    Returns
    -------
    Dict[str, pd.Series]
        资产名 → 收益率 Series (DatetimeIndex)。
    """
    df = pd.read_excel(input_path, sheet_name=sheet)
    df["Date"] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index("Date")

    assets = {}
    for name_cn, col_idx in ASSET_COL_INDEX.items():
        # col 0 = Date, assets start at col 1
        col_name = df.columns[col_idx - 1]
        assets[name_cn] = df[col_name]
    return assets


def split_periods(
    assets: Dict[str, pd.Series],
    cutoffs: Optional[List[str]] = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    将各资产数据按时间段切分。

    Parameters
    ----------
    assets : Dict[str, pd.Series]
    cutoffs : Optional[List[str]]
        划分时间点，默认 ["2020-12-31"]。

    Returns
    -------
    dict: assets[name][period_label] = series
    """
    if cutoffs is None:
        cutoffs = ["2020-12-31"]

    periods: Dict[str, Dict[str, pd.Series]] = {}
    for name_cn, series in assets.items():
        periods[name_cn] = {"full": series}
        for i, co in enumerate(cutoffs):
            label = f"pre_{co}" if i == 0 else f"mid_{co}"
            periods[name_cn][label] = series[series.index <= co]
        # 最后一个 cutoff 之后的时段
        periods[name_cn]["post"] = series[series.index >= cutoffs[-1]]
    return periods


# ──────────────────────────────────────────────
# 完整分析管线
# ──────────────────────────────────────────────
TrendMetrics = Dict[str, Dict]  # {period_label: {metric_name: value}}


def analyze_all(
    assets: Dict[str, pd.Series],
    period_data: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对所有资产执行完整趋势分析。

    Returns
    -------
    (metric_detail_df, ma_backtest_df)
    """
    if period_data is None:
        period_data = split_periods(assets)

    stat_metrics = {
        "SNR": compute_snr,
        "Hurst": compute_hurst,
        "Trend_Stability": compute_trend_stability,
        "Trend_Efficiency": lambda s: compute_trend_efficiency(s)[
            "trend_efficiency"
        ],
        "Trend_Strength": lambda s: compute_trend_efficiency(s)["trend_strength"],
    }

    rows_stat, rows_ac, rows_streak, rows_ma = [], [], [], []

    for name_cn in assets:
        periods_dict = period_data[name_cn]
        for period_label, series in periods_dict.items():
            if len(series) < 100:
                continue

            # 统计指标
            row = {
                "asset": name_cn,
                "asset_en": ASSET_NAMES_EN.get(name_cn, name_cn),
                "period": period_label,
                "n_obs": len(series),
            }
            for mname, fn in stat_metrics.items():
                row[mname] = fn(series)
            rows_stat.append(row)

            # 自相关
            ac = compute_autocorr(series)
            row_ac = {
                "asset": name_cn,
                "asset_en": ASSET_NAMES_EN.get(name_cn, name_cn),
                "period": period_label,
            }
            row_ac.update(ac)
            rows_ac.append(row_ac)

            # 连涨连跌
            streak = compute_streaks(series)
            row_streak = {
                "asset": name_cn,
                "asset_en": ASSET_NAMES_EN.get(name_cn, name_cn),
                "period": period_label,
            }
            row_streak.update(streak)
            rows_streak.append(row_streak)

            # 双均线回测
            ma = compute_ma_backtest(series)
            row_ma = {
                "asset": name_cn,
                "asset_en": ASSET_NAMES_EN.get(name_cn, name_cn),
                "period": period_label,
            }
            row_ma.update(ma)
            rows_ma.append(row_ma)

    df_stat = pd.DataFrame(rows_stat)
    df_ac = pd.DataFrame(rows_ac)
    df_streak = pd.DataFrame(rows_streak)
    df_ma = pd.DataFrame(rows_ma)

    df_detail = (
        df_stat.merge(df_ac, on=["asset", "asset_en", "period"])
        .merge(df_streak, on=["asset", "asset_en", "period"])
    )
    return df_detail, df_ma


# ──────────────────────────────────────────────
# 排名管线
# ──────────────────────────────────────────────
RANK_DIRECTION: Dict[str, bool] = {
    "SNR": True,
    "Hurst": True,
    "Trend_Stability": True,
    "Trend_Efficiency": True,
    "Trend_Strength": True,
    "AC(1)": True,
    "AC(5)": True,
    "AC(20)": True,
    "max_consec_up": True,
    "max_consec_down": True,
    "ann_return": True,
    "sharpe": True,
    "max_drawdown": False,
    "win_rate": True,
    "calmar": True,
}

WEIGHTS: Dict[str, float] = {
    "SNR": 0.15,
    "Hurst": 0.15,
    "sharpe": 0.15,
    "Trend_Stability": 0.12,
    "Trend_Efficiency": 0.12,
    "win_rate": 0.05,
    "max_drawdown": 0.08,
    "calmar": 0.05,
    "AC(1)": 0.05,
    "AC(5)": 0.03,
    "AC(20)": 0.03,
    "ann_return": 0.02,
}

MERGE_COLS = ["asset", "asset_en", "period"]


def compute_rankings(
    df_detail: pd.DataFrame,
    df_ma: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    按时间段分别计算各资产的综合排名。

    Returns
    -------
    Dict[period_label, ranking_df]
    """
    df_full = df_detail.merge(
        df_ma[MERGE_COLS + ["ann_return", "sharpe", "max_drawdown", "win_rate", "calmar", "num_trades"]],
        on=MERGE_COLS,
        how="left",
    )

    ranking_sheets = {}
    for period in df_full["period"].unique():
        pdf = df_full[df_full["period"] == period].copy()
        if pdf.empty:
            continue

        rank_dir = {k: v for k, v in RANK_DIRECTION.items() if k in pdf.columns}
        rank_df = cross_sectional_rank(
            pdf.set_index("asset")[list(rank_dir.keys())],
            rank_dir,
        )
        weight_cols = [m for m in WEIGHTS if f"{m}_rank" in rank_df.columns]
        w = {m: WEIGHTS[m] for m in weight_cols}
        scored = composite_score(rank_df, w)

        scored = scored.reset_index().merge(
            pdf[MERGE_COLS], left_on="asset", right_on="asset", how="left"
        )
        scored = scored.sort_values("overall_rank")
        ranking_sheets[period] = scored

    return ranking_sheets


# ──────────────────────────────────────────────
# 导出
# ──────────────────────────────────────────────
def export_excel(
    df_detail: pd.DataFrame,
    df_ma: pd.DataFrame,
    rankings: Dict[str, pd.DataFrame],
    output_path: str,
):
    """写入 Excel"""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_detail.to_excel(writer, sheet_name="metric_detail", index=False)
        df_ma.to_excel(writer, sheet_name="ma_backtest", index=False)
        for sheet_name, rdf in rankings.items():
            rdf.to_excel(writer, sheet_name=f"rank_{sheet_name}", index=False)
    print(f"  → {output_path}")


def main():
    """CLI 入口"""
    import sys

    # 默认值，可通过命令行覆盖
    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "..", "allCycleInput202605.xlsx")
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(input_file)

    print("=" * 60)
    print("趋势跟踪适合度定量分析")
    print("=" * 60)

    print("\nStep 1/4: 读取数据...")
    assets = load_returns(input_file)
    period_data = split_periods(assets)

    print("\nStep 2/4: 计算指标...")
    df_detail, df_ma = analyze_all(assets, period_data)
    print(f"  {len(df_detail)} 个资产-时段组合")

    print("\nStep 3/4: 排名...")
    rankings = compute_rankings(df_detail, df_ma)
    for k, v in rankings.items():
        print(f"  {k}: {len(v)} 行")

    print("\nStep 4/4: 导出...")
    output = os.path.join(output_dir, "trend_following_output.xlsx")
    export_excel(df_detail, df_ma, rankings, output)

    print("\n分析完成！")


if __name__ == "__main__":
    main()
