"""
案例: 从 Wind EDB 提取宏观因子 → 生成 DataFrame → 衍生计算

数据源: config/macro_tickers.csv 中定义的所有指标
流程:
  1. WindFetcher 读取 CSV 配置
  2. 逐个序列从 Wind 取数 (或读 Parquet 缓存)
  3. 合并为月频 DataFrame
  4. 计算衍生指标 (M1-M2 剪刀差、社融脉冲、信用因子)

运行:
  # 有 Wind 终端: 实时拉取
  python demo_macro_wind_to_cycle.py

  # 无 Wind: 展示 CSV 配置和缓存状态
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _wind_available() -> bool:
    try:
        from WindPy import w
        w.start()
        return w.isconnected()
    except Exception:
        return False


def show_config():
    """展示 CSV 配置"""
    from models.data_loader import read_asset_config, read_macro_config

    asset_cfg = read_asset_config()
    macro_cfg = read_macro_config()

    print("=" * 72)
    print("  配置: config/asset_tickers.csv")
    print("=" * 72)
    print(asset_cfg[["name", "ticker", "type", "freq"]].to_string(index=False))

    print()
    print("=" * 72)
    print("  配置: config/macro_tickers.csv")
    print("=" * 72)
    print(macro_cfg[["name", "ticker", "freq", "group", "desc"]].to_string(index=False))


def show_cache_status():
    """展示 Parquet 缓存状态"""
    from models.data_loader import WindFetcher

    fetcher = WindFetcher()
    cache_info = fetcher.list_cache()

    if len(cache_info) == 0:
        print("\n  📭 无 Parquet 缓存")
    else:
        print("\n" + "=" * 72)
        print("  Parquet 缓存状态")
        print("=" * 72)
        print(cache_info[["name", "rows", "date_range", "size_kb"]].to_string(index=False))


def compute_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算衍生指标:
      (1) M1-M2 剪刀差 + 3M/12M 移动平均
      (2) 社融脉冲 (SF_ratio 3 个月变化方向)
      (3) 信用因子信号
    """
    result = df.copy()

    # (1) M1-M2 剪刀差
    if "M1_YoY" in result.columns and "M2_YoY" in result.columns:
        result["M1-M2_spread"] = result["M1_YoY"] - result["M2_YoY"]
        result["M1-M2:MA3"] = result["M1-M2_spread"].rolling(3).mean()
        result["M1-M2:MA12"] = result["M1-M2_spread"].rolling(12).mean()
        result["M1-M2:DMA"] = result["M1-M2:MA3"] - result["M1-M2:MA12"]
        print("  ✅ M1-M2 剪刀差计算完成")
        if not result["M1-M2:MA3"].isna().all():
            print(f"     最近 3M 均值: {result['M1-M2:MA3'].iloc[-1]:.2f}%")

    # (2) 信用因子: 社融同比 → 3 个月变化方向
    if "SF_ratio" in result.columns:
        result["credit_pulse"] = result["SF_ratio"].diff(3)
        result["credit_signal"] = np.where(
            result["credit_pulse"].notna(),
            np.sign(result["credit_pulse"]),
            0,
        )
        print("  ✅ 信用因子 (credit_signal) 计算完成")

    # (3) 社融脉冲
    if "SF_total" in result.columns:
        sf_diff = result["SF_total"].diff()
        sf_rolling = sf_diff.rolling(12).sum()
        sf_yoy = sf_rolling.pct_change(12) * 100
        result["sf_pulse"] = sf_yoy
        result["sf_pulse_3m"] = sf_yoy.diff(3)
        print("  ✅ 社融脉冲 (sf_pulse) 计算完成")

    return result


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   宏观因子 Wind EDB → DataFrame → 衍生指标            ║")
    print("║   CSV 驱动 + Parquet 缓存                              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Step 0: 展示配置
    show_config()

    # Step 1: 检查缓存
    show_cache_status()

    if _wind_available():
        # Step 2: 从 Wind 取数
        print("\n" + "=" * 72)
        print("  从 Wind EDB 提取数据")
        print("=" * 72)

        from models.data_loader import WindFetcher
        fetcher = WindFetcher()
        df_macro = fetcher.fetch_macro(use_cache=True)

        print(f"\n  📊 宏观数据: {df_macro.shape[0]} 行 × {df_macro.shape[1]} 列")
        print(f"  时间: {df_macro['Date'].min():%Y-%m-%d} ~ {df_macro['Date'].max():%Y-%m-%d}")
        print()
        print("  列:", list(df_macro.columns))
        print()
        print("  前 8 行:")
        print(df_macro.head(8).to_string(index=False))
        print()
        print("  后 5 行:")
        print(df_macro.tail(5).to_string(index=False))

        # Step 3: 衍生计算
        print("\n" + "─" * 72)
        print("  衍生指标计算")
        print("─" * 72)
        df_derived = compute_derived_indicators(df_macro)

        derived_cols = [c for c in df_derived.columns
                       if c not in ("Date", "CN10Y", "US10Y", "CPI", "PMI")]
        print()
        print(df_derived[["Date"] + derived_cols].tail(8).to_string(index=False))

    else:
        print("\n  ⚠️  WindPy 未连接，进入演示模式")
        print("  启动 Wind 终端后，脚本将自动:")
        print("    1. 读取 CSV 配置中的 Wind 代码")
        print("    2. 逐个调用 wsd/edb 取数")
        print("    3. 存储为 Parquet 缓存 (data/cache/)")
        print("    4. 合并为月频 DataFrame")
        print("    5. 计算 M1-M2 剪刀差 / 社融脉冲 / 信用因子")
        print()
        print("  新增数据源: 在 config/macro_tickers.csv 加一行即可")
        print("  删除数据源: 删掉那行即可")

    print()
    print("=" * 72)
    print("  ✅ 完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
