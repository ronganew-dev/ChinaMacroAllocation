"""
一键更新 allCycleInput.xlsx — 通过 Wind API 增量更新

使用 models.data_loader.WindDataLoader 取代手写 Wind API 调用，
资产映射从 data_hub_config.json 统一读取。

流程：
  1. 备份原文件
  2. 增量拉取 Wind ETF/指数收益率 → 写入 'return' 工作表
  3. 拉取 CN10Y/US10Y 日度数据 → 写入 'dailyYield' 工作表
  4. 聚合月频宏观因子 → 更新 'macroData' 工作表
  5. 保存 Excel
"""

import datetime
import json
import os
import shutil

import numpy as np
import pandas as pd

from models.data_loader import WindDataLoader
from models.universe import Universe


# ── 路径 ──

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "data_hub_config.json")
FILE_PATH = os.path.join(SCRIPT_DIR, "data", "input", "allCycleInput.xlsx")
BACKUP_DIR = os.path.join(SCRIPT_DIR, "backup")


# ── 工具函数 ──

def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _backup():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(FILE_PATH))
    shutil.copy(FILE_PATH, backup_path)
    print(f"[*] 备份完成: {FILE_PATH} → {backup_path}")


def _read_sheets() -> dict:
    print(f"[*] 读取 {FILE_PATH} 中的所有数据表...")
    return pd.read_excel(FILE_PATH, sheet_name=None)


def _write_sheets(sheets: dict):
    for name, df in sheets.items():
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            print(f"    工作表 '{name}' 已存入 Excel (共 {len(df)} 行).")


def _get_asset_names(config: dict) -> list:
    """从配置中获取完整的资产名称列表（含安全资产）"""
    universe_cfg = config.get("universe", {})
    risky = universe_cfg.get("risky_assets", [])
    safe = [universe_cfg.get("safe_asset", "中债新综合财富总值指数")]
    # 补充其他可能在 Excel 中存在但不在 risky_assets 中的资产
    extra = [
        "中债新综合财富总值指数", "黄金", "有色",
        "量化中性", "信用债3-5年", "信用债0-1年",
        "国债30年", "信用债1-3年", "中证商品指数",
        "现金", "自由现金流",
    ]
    all_assets = list(dict.fromkeys(risky + safe + extra))  # 去重保序
    return all_assets


# ── 主函数 ──

def update_all_data():
    print("==================================================")
    print("  启动: Wind API 一键更新收益率与宏观因子数据")
    print("==================================================")

    config = _load_config()
    today = datetime.datetime.now()
    end_date = today.strftime("%Y-%m-%d")

    # ── 1. 备份 ──
    _backup()

    # ── 2. 读取现有 Excel ──
    sheets = _read_sheets()

    # =============================================================
    # 模块一：更新 'return' 日频收益率
    # =============================================================
    print("\n>>>> [1/3] 正在更新 'return' (收益率工作表) ...")

    df_ret = sheets.get("return")
    if df_ret is None or "Date" not in df_ret.columns:
        print("错误: 未找到 'return' 工作表或其 'Date' 列!")
        return

    df_ret["Date"] = pd.to_datetime(df_ret["Date"])
    start_date_ret = df_ret["Date"].max().strftime("%Y-%m-%d")
    print(f"    本地最新记录日期: {start_date_ret}, 目标终点: {end_date}")

    # 通过 WindDataLoader 增量获取收益率
    loader = WindDataLoader(config)
    all_assets = _get_asset_names(config)
    df_new = loader.load_returns(all_assets, start_date=start_date_ret, end_date=end_date)

    # 只保留增量部分
    df_new = df_new[df_new["Date"] > pd.to_datetime(start_date_ret)]

    if len(df_new) > 0:
        df_combined = pd.concat([df_ret, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset="Date", keep="last")
        df_combined = df_combined.sort_values("Date").reset_index(drop=True)
        sheets["return"] = df_combined
        print(f"    [成功] 'return' 表新增 {len(df_new)} 行日频收益率数据.")
    else:
        print("    [完成] 'return' 已经是最新数据，无新增行.")

    # =============================================================
    # 模块二：CN10Y / US10Y 日度数据 → dailyYield
    # =============================================================
    print("\n>>>> [2/3] 正在更新 'dailyYield' (日度美债/中债收益率) ...")

    macro_mappings = config.get("macro_mappings", {})
    df_yields = loader._fetch_yields(macro_mappings, "2011-01-01", end_date)
    sheets["dailyYield"] = df_yields
    print(f"    [成功] 'dailyYield' 工作表生成完毕，共 {len(df_yields)} 行日频数据.")

    # =============================================================
    # 模块三：更新 'macroData' 月频宏观因子
    # =============================================================
    print("\n>>>> [3/3] 正在更新 'macroData' (月度宏观因子工作表) ...")

    df_macro = sheets.get("macroData")
    if df_macro is None or "Date" not in df_macro.columns:
        print("错误: 未找到 'macroData' 工作表或其 'Date' 列!")
        return

    df_macro["Date"] = pd.to_datetime(df_macro["Date"])
    max_macro_date = df_macro["Date"].max()

    # 补全缺失月份行（保留社融 ffill，其余为 NaN）
    new_months = pd.date_range(
        start=max_macro_date + pd.Timedelta(days=1), end=today, freq="ME"
    )
    if len(new_months) > 0:
        last = df_macro.iloc[-1].to_dict()
        new_rows = []
        for dt in new_months:
            row = {"Date": dt}
            row["SF_ratio"] = last.get("SF_ratio", np.nan)
            row["SF_total"] = last.get("SF_total", np.nan)
            for col in ["CPI", "PMI", "CN10Y", "US10Y"]:
                row[col] = np.nan
            new_rows.append(row)
        df_macro = pd.concat([df_macro, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"    补全至最新月份共 {len(new_months)} 个月空行.")

    df_macro["YM"] = df_macro["Date"].dt.to_period("M")
    df_macro = df_macro.set_index("YM")

    # 从 dailyYield 聚合月均值
    df_yields_local = df_yields.copy()
    df_yields_local["YM"] = pd.to_datetime(df_yields_local["Date"]).dt.to_period("M")
    monthly_avg = df_yields_local.groupby("YM")[["CN10Y", "US10Y"]].mean()
    df_macro.update(monthly_avg)
    print("    [更新] CN10Y / US10Y 月度均值已填入.")

    # 从 Wind 拉取 CPI / PMI / 社融月度数据
    df_wind_m = loader._fetch_monthly_macro(macro_mappings, "2011-01-01", end_date)
    df_macro.update(df_wind_m)

    df_sf = loader._fetch_social_financing(macro_mappings, "2011-01-01", end_date)
    if df_sf is not None:
        df_macro.update(df_sf)

    print("    [更新] CPI / PMI / 社融 最新月度数据已填入.")

    # 填充空值并存回 sheets
    df_macro = df_macro.reset_index()
    for col in ["CPI", "PMI", "CN10Y", "US10Y", "SF_ratio", "SF_total"]:
        if col in df_macro.columns:
            df_macro[col] = df_macro[col].ffill().bfill()
    if "YM" in df_macro.columns:
        df_macro = df_macro.drop(columns=["YM"])
    sheets["macroData"] = df_macro
    print("    [ffill] 宏观因子空值行填充完毕.")

    # =============================================================
    # 写入 Excel
    # =============================================================
    print("\n>>>> [写入] 格式化 Date 列并保存工作簿 ...")
    _write_sheets(sheets)

    print("\n==================================================")
    print("  [完成] 数据一键更新全部成功!")
    print("==================================================")


if __name__ == "__main__":
    update_all_data()
