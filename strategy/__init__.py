"""
strategy — 宏观周期资产配置策略

包含中国宏观经济周期信号驱动的多资产配置策略，
支持 EWMA 风险预算、子组合风险控制和动态再平衡。
"""

from strategy.config import Config
from strategy.utils import (
    calc_ewma,
    find_month_end,
    write_dataframes_to_excel,
)

__all__ = [
    "Config",
    "calc_ewma",
    "find_month_end",
    "write_dataframes_to_excel",
]
