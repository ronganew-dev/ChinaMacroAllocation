"""
一键更新 allCycleInput.xlsx — 通过 Wind API 增量更新

使用 WindFetcher 从 Wind 取数，CSV 配置驱动，Parquet 缓存。
数据映射来自 config/asset_tickers.csv 和 config/macro_tickers.csv。

流程:
  1. 备份原 Excel
  2. WindFetcher 取资产收益率 → 更新 'return' 工作表
  3. WindFetcher 取 CN10Y/US10Y → 更新 'dailyYield' 工作表
  4. WindFetcher 取宏观指标 → 更新 'macroData' 工作表
  5. 保存 Excel
"""

import datetime
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from models.data_loader import WindFetcher


SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR / "data" / "input" / "allCycleInput.xlsx"
BACKUP_DIR = SCRIPT_DIR / "backup"


def _backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / FILE_PATH.name
    shutil.copy(FILE_PATH, backup_path)
    print(f"[*] 备份完成: {FILE_PATH} → {backup_path}")


def _read_sheets() -> dict:
    print(f"[*] 读取 {FILE_PATH} ...")
    return pd.read_excel(FILE_PATH, sheet_name=None)


def _write_sheets(sheets: dict):
    for name, df in sheets.items():
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    with pd.ExcelWriter(str(FILE_PATH), engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            print(f"    工作表 '{name}' 已存入 (共 {len(df)} 行)")


def update_all_data():
    print("=" * 50)
    print("  Wind API 一键更新收益率与宏观因子数据")
    print("=" * 50)

    today = datetime.datetime.now()
    end_date = today.strftime("%Y-%m-%d")

    # 1. 备份
    _backup()

    # 2. 读取 Excel
    sheets = _read_sheets()

    # 3. 实例化 WindFetcher
    fetcher = WindFetcher()

    # =============================================================
    # 模块一: 更新 'return' 日频收益率
    # =============================================================
    print("\n>>>> [1/3] 更新 'return' (收益率工作表) ...")

    df_ret = sheets.get("return")
    if df_ret is None or "Date" not in df_ret.columns:
        print("错误: 未找到 'return' 工作表")
        return

    df_ret["Date"] = pd.to_datetime(df_ret["Date"])

    # 从 CSV 配置获取全部资产名
    all_assets = fetcher.asset_cfg["name"].tolist()
    df_new = fetcher.fetch_returns(asset_names=all_assets, use_cache=True)

    if df_new is not None and len(df_new) > 0:
        df_new["Date"] = pd.to_datetime(df_new["Date"])
        df_combined = pd.concat([df_ret, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset="Date", keep="last")
        df_combined = df_combined.sort_values("Date").reset_index(drop=True)
        sheets["return"] = df_combined
        print(f"    [成功] 'return' 表已更新")
    else:
        print("    [完成] 'return' 已是最新")

    # =============================================================
    # 模块二: CN10Y / US10Y → dailyYield
    # =============================================================
    print("\n>>>> [2/3] 更新 'dailyYield' (日度收益率) ...")

    df_cn10y = fetcher.fetch_series("CN10Y", use_cache=True)
    df_us10y = fetcher.fetch_series("US10Y", use_cache=True)

    if df_cn10y is not None and df_us10y is not None:
        df_daily = pd.merge(df_cn10y, df_us10y, on="Date", how="outer")
        df_daily = df_daily.sort_values("Date").reset_index(drop=True)
        sheets["dailyYield"] = df_daily
        print(f"    [成功] 'dailyYield' 已生成 ({len(df_daily)} 行)")
    else:
        print("    ⚠️ 收益率数据获取失败")

    # =============================================================
    # 模块三: 更新 'macroData' 月频宏观因子
    # =============================================================
    print("\n>>>> [3/3] 更新 'macroData' (月度宏观因子) ...")

    df_macro_new = fetcher.fetch_macro(use_cache=True)

    if df_macro_new is not None and len(df_macro_new) > 0:
        # 填充空值
        value_cols = [c for c in df_macro_new.columns if c != "Date"]
        df_macro_new[value_cols] = df_macro_new[value_cols].ffill().bfill()
        sheets["macroData"] = df_macro_new
        print(f"    [成功] 'macroData' 已更新 ({len(df_macro_new)} 行)")
    else:
        print("    ⚠️ 宏观数据获取失败")

    # 4. 保存
    _write_sheets(sheets)
    print("\n" + "=" * 50)
    print("  ✅ 全部更新完成")
    print("=" * 50)


if __name__ == "__main__":
    update_all_data()
