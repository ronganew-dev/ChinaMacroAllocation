"""
日历/日期工具类算子 — Formulaic Operators

每个函数 = 一个原子公式，纯向量化，零显式循环。
输入 pd.DatetimeIndex / Dict → 输出 pd.Series / None。

"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

# ===================================================================
# 月末切换标志 | Month-End Boundary Detection
# ===================================================================

def find_month_end(dates: pd.DatetimeIndex) -> pd.Series:
    r"""
    标记月份切换日 — 全向量化，零显式循环。

    返回一个布尔 Series，在每个月第一天（即从上一个月的最后一天
    转换到新月份的日期）标记为 True。

    Formula:
        flag_t = True   if month_t ≠ month_{t-1} or year_t ≠ year_{t-1}
                 True   if t = 0
                 False  otherwise

    Parameters
    ----------
    dates : pd.DatetimeIndex
        时间索引序列。

    Returns
    -------
    pd.Series of bool
        与 dates 等长，True 表示该日期为月份切换日。
        首个元素恒为 True。

    Examples
    --------
    >>> dates = pd.date_range("2026-01-31", periods=4, freq="D")
    >>> find_month_end(dates)
    2026-01-31    True
    2026-02-01    True
    2026-02-02   False
    2026-02-03   False
    dtype: bool
    """
    month = pd.Series(dates.month, index=dates)
    year = pd.Series(dates.year, index=dates)

    # 全向量化：检测月份或年份是否发生变化
    changed = (month != month.shift(1)) | (year != year.shift(1))
    changed.iloc[0] = True

    return changed


# ===================================================================
# DataFrame 批量写入 Excel | Multi-Sheet Excel Writer
# ===================================================================

def write_dataframes_to_excel(
    df_dict: Dict[str, pd.DataFrame],
    file_path: str,
    engine: str = "openpyxl",
) -> None:
    r"""
    将多个 DataFrame 写入 Excel 文件的不同 Sheet。

    若目标文件已存在，以追加模式写入（同名 Sheet 会被替换）；
    若不存在，创建新文件。

    此函数为 I/O 工具，严格遵循 Formulaic Operators 的
    纯函数式接口约定（无类、无状态、无副作用声明）。

    Parameters
    ----------
    df_dict : Dict[str, pd.DataFrame]
        {sheet_name: DataFrame} 映射。
    file_path : str
        输出 Excel 文件路径。
    engine : str, default "openpyxl"
        pandas ExcelWriter 引擎，仅影响 xlsx 文件。

    Returns
    -------
    None
        直接写入文件系统。

    Examples
    --------
    >>> write_dataframes_to_excel(
    ...     {"returns": df_ret, "signals": df_sig},
    ...     "./output.xlsx",
    ... )
    """
    import os

    if os.path.isfile(file_path):
        with pd.ExcelWriter(
            file_path,
            engine=engine,
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(file_path, engine=engine) as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
