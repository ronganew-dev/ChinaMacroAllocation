"""
宏观周期配置策略 — 全局参数配置

从 configAllCycle.py 重构提取，保持原语义。
"""

import numpy as np


class Config:
    """策略全局配置"""

    # ---- 文件路径 ----
    input_folder = "./"
    output_folder = "./"
    input_file = input_folder + "allCycleInput.xlsx"
    output_file = output_folder + "allCycleOutput.xlsx"

    return_sheet = "return"
    sub_port_sheet = "subPort"
    daa_prob_sheet = "DAAProb"
    signal_sheet = "signal"

    # ---- 资产定义 ----
    risky_asset_names = [
        "沪深300",
        "标普500",
        "量化中性",
        "黄金",
        "中证商品指数",
        "国债30年",
        "信用债1-3年",
    ]
    safe_asset_names = ["信用债0-1年", "现金"]
    asset_names = risky_asset_names + safe_asset_names
    regime_names = {"过热", "复苏", "滞胀", "衰退"}
    start_date = "2017/1/1"

    # ---- 时间参数 ----
    days_per_year = 365
    b_days_per_year = 252
    months_per_year = 12

    # ---- EWMA ————
    ewma_halflife_days = 63
    ewma_decay_factor = 0.5 ** (1 / ewma_halflife_days)
    ewma_rolling_days = 1260

    # ---- 风险预算 ————
    risk_scaler = 0.1
    target_risk = 0.02
    ex_ante_risk_up_bound = 0.08

    # ---- 调仓 ----
    month_end_rebalance = False
    daily_no_tcost_rebalance = True
    trade_size = 1.0
    threshold = [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.20,
        0.20,
    ]
    trade_delay = 1
    transaction_cost = [
        0.0002,
        0.0002,
        0.0002,
        0.0002,
        0.0002,
        0.0002,
        0.0002,
        0.0001,
        0.0001,
    ]

    # ---- 经济状态 ----
    regime_column = "regime"
    sub_port_target_risk = 0.02
    cash_weight = 0.05

    # ---- 量化中性调整 ————
    adjusted_asset_name = "量化中性"
    adjusted_asset_mean = 0.055
    vol_multiplier = 1.5
    performance_fee = 0.2
    return_adjustment = (
        adjusted_asset_mean
        * (vol_multiplier - 1 + performance_fee)
        / b_days_per_year
    )
