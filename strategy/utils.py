"""
宏观周期配置策略 — 工具函数

从 utils.py 重构提取，EWMA 协方差和回撤计算委托给 operators。
"""

import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from operators.volatility import ewma_cov

# 为了向后兼容
calc_ewma = ewma_cov


def find_month_end(dates: pd.DatetimeIndex) -> list:
    """标记月份切换日"""
    month_end = []
    for i in range(len(dates)):
        if i == 0:
            month_end.append(True)
        else:
            month_end.append(
                dates[i].month != dates[i - 1].month
                or dates[i].year != dates[i - 1].year
            )
    return month_end


def write_dataframes_to_excel(df_dict: dict, file_path: str):
    """将多个 DataFrame 写入 Excel 文件的不同 sheet"""
    if os.path.isfile(file_path):
        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
