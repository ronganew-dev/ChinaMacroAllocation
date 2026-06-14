"""
案例：从 Wind EDB 提取宏观因子 → 生成 DataFrame → 进入六周期模型

数据源
------
M1 同比 (M0000551)    社会融资规模存量 (M5525546)
M2 同比 (M0001385)    社会融资规模存量同比 (M5525547)
CPI 同比 (M0000612)   中国10年国债收益率 (M0325687)
PMI (M0017126)        美国10年国债收益率 (G0000891)

流程
----
  1. 通过 WindDataLoader 从 Wind EDB 拉取原始宏观数据
  2. 组装为月度 DataFrame（含 M1/M2/社融/CPI/PMI/CN10Y/US10Y）
  3. 计算衍生指标：M1-M2 剪刀差、社融脉冲、中长期贷款脉冲
  4. 调用 Cycle6 模型生成货币/信用/增长因子
  5. 输出六周期划分结果（表格 + 统计）

前提
----
  - Wind 终端已启动 (WindPy 可用)
  - 如无 Wind 环境，脚本自动回退为演示模式，从 data_hub_config.json 展示元数据
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. 路径与配置
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "data_hub_config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

MACRO_MAPPINGS = CONFIG["macro_mappings"]

CYCLE_MAP = {
    1: "信用扩张",
    2: "经济复苏",
    3: "货币退潮",
    4: "信用退潮",
    5: "经济放缓",
    6: "货币扩张",
}
CYCLE_COLORS = {
    1: "🟢",
    2: "🟩",
    3: "🟡",
    4: "🟠",
    5: "🔴",
    6: "🟤",
}


# ─────────────────────────────────────────────────────────────────────
# 2. 数据提取：Wind EDB → 原始 DataFrame
# ─────────────────────────────────────────────────────────────────────

def _wind_available() -> bool:
    """检查 WindPy 是否可用且已连接"""
    try:
        from WindPy import w
        w.start()
        return w.isconnected()
    except Exception:
        return False


def show_macro_mappings():
    """打印当前宏观因子映射表"""
    print("=" * 72)
    print("  📋 宏观因子 Wind EDB 映射表")
    print("=" * 72)
    for name, info in MACRO_MAPPINGS.items():
        print(f"    {name:12s} → {info['ticker']}")
    print()


def fetch_all_from_wind(start="2011-01-01", end=None) -> pd.DataFrame:
    """
    通过 WindDataLoader 拉取所有宏观因子数据。
    返回月频 DataFrame。
    """
    from models.data_loader import WindDataLoader

    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"  📡 数据区间: {start} ~ {end}")
    loader = WindDataLoader(CONFIG)

    # ── 拉取全部原始数据 ──
    print("  [1/5] 拉取 CN10Y / US10Y 日度收益率 ...")
    df_yields = loader._fetch_yields(MACRO_MAPPINGS, start, end)
    df_yields["YM"] = pd.to_datetime(df_yields["Date"]).dt.to_period("M")
    monthly_yield_avg = df_yields.groupby("YM")[["CN10Y", "US10Y"]].mean()
    print(f"         → {len(monthly_yield_avg)} 个月度观测")

    print("  [2/5] 拉取 CPI / PMI 月度数据 ...")
    df_cpi_pmi = loader._fetch_monthly_macro(MACRO_MAPPINGS, start, end)
    print(f"         → {len(df_cpi_pmi)} 条记录")

    print("  [3/5] 拉取社会融资总规模 ...")
    df_sf = loader._fetch_social_financing(MACRO_MAPPINGS, start, end)
    if df_sf is not None:
        print(f"         → {len(df_sf)} 条记录 (SF_ratio, SF_total)")
    else:
        print("         → ⚠️ 社融数据不可用")

    print("  [4/5] 拉取 M1 / M2 同比增速 ...")
    df_m1m2 = loader._fetch_m1m2(MACRO_MAPPINGS, start, end)
    if df_m1m2 is not None:
        print(f"         → {len(df_m1m2)} 条记录 (M1_YoY, M2_YoY)")
    else:
        print("         → ⚠️ M1/M2 数据不可用")

    print("  [5/5] 组装统一月度 DataFrame ...")

    # ── 统一时间索引 ──
    monthly_range = pd.date_range(
        start=pd.to_datetime(start), end=pd.to_datetime(end), freq="ME"
    )
    df = pd.DataFrame(index=monthly_range)
    df.index.name = "Date"
    df = df.reset_index()
    df["YM"] = df["Date"].dt.to_period("M")
    df = df.set_index("YM")

    # 填入数据（位置尽可能靠前，方便后续核对）
    df = df.join(monthly_yield_avg, how="left")
    df = df.join(df_cpi_pmi, how="left")
    if df_sf is not None:
        df = df.join(df_sf, how="left")
    if df_m1m2 is not None:
        df = df.join(df_m1m2, how="left")

    df = df.reset_index(drop=True).ffill().bfill()
    return df


def print_dataframe_snapshot(df: pd.DataFrame, title: str = "宏观因子 DataFrame"):
    """打印 DataFrame 概况"""
    print()
    print("=" * 72)
    print(f"  📊 {title}")
    print("=" * 72)
    print(f"  维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  列:   {list(df.columns)}")
    print(f"  时间: {df['Date'].min():%Y-%m-%d} ~ {df['Date'].max():%Y-%m-%d}")
    print()
    print("  前 8 行:")
    print(df.head(8).to_string(index=False))
    print()
    print("  后 5 行:")
    print(df.tail(5).to_string(index=False))
    print()
    print("  描述统计:")
    print(df.describe().round(4).to_string())
    print()


# ─────────────────────────────────────────────────────────────────────
# 3. 衍生指标计算
# ─────────────────────────────────────────────────────────────────────

def compute_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    从原始宏观数据计算衍生指标：

    (1) 社融存量同比 → Δ₃M 方向打分 → 信用因子信号
    (2) M1-M2 剪刀差 → 3M/12M 移动平均 → DMA
    (3) 中长期贷款脉冲（社融替代）→ Δ₁₂M → 同比 → Δ₃M
    """
    result = df.copy()

    # ── (1) 信用因子：社融存量同比 → 3个月环比方向 ──
    if "SF_ratio" in result.columns:
        # SF_ratio 本身就是同比增速，直接看 3 个月变化方向
        result["credit_pulse"] = result["SF_ratio"].diff(3)
        result["credit_signal"] = np.where(
            result["credit_pulse"].notna(),
            np.sign(result["credit_pulse"]),
            0,
        )
        print("  ✅ 信用因子 (credit_signal): 取自社融存量同比的 3 个月环比方向")
    else:
        result["credit_pulse"] = 0.0
        result["credit_signal"] = 0
        print("  ⚠️ 信用因子: 无社融数据，默认设为 0")

    # ── (2) M1-M2 剪刀差 ──
    if "M1_YoY" in result.columns and "M2_YoY" in result.columns:
        result["M1-M2_spread"] = result["M1_YoY"] - result["M2_YoY"]
        result["M1-M2:MA3"] = result["M1-M2_spread"].rolling(3).mean()
        result["M1-M2:MA12"] = result["M1-M2_spread"].rolling(12).mean()
        result["M1-M2:DMA"] = result["M1-M2:MA3"] - result["M1-M2:MA12"]
        print("  ✅ M1-M2 剪刀差及滚动均值计算完成")
        print(f"     最近 3M 均值: {result['M1-M2:MA3'].iloc[-1]:.2f}%")
        print(f"     最近 12M 均值: {result['M1-M2:MA12'].iloc[-1]:.2f}%")
        print(f"     DMA: {result['M1-M2:DMA'].iloc[-1]:.2f}%")
    else:
        print("  ⚠️ M1/M2 数据缺失，跳过剪刀差计算")

    # ── (3) 中长期贷款脉冲（用社融存量模拟） ──
    if "SF_total" in result.columns:
        # 社融存量 → 月度差分 → 12M 滚动求和 → 同比 → 3M 环比
        sf_diff = result["SF_total"].diff()  # 月度增量
        sf_rolling = sf_diff.rolling(12).sum()  # 12M 滚动求和
        sf_yoy = sf_rolling.pct_change(12) * 100  # 同比
        result["sf_pulse"] = sf_yoy
        result["sf_pulse_3m"] = sf_yoy.diff(3)
        print("  ✅ 社融脉冲指标 (sf_pulse) 计算完成")
    else:
        print("  ⚠️ 社融存量数据缺失，跳过脉冲计算")

    return result


def print_derived_snapshot(df: pd.DataFrame):
    """打印衍生指标的最新值"""
    print()
    print("─" * 72)
    print("  📈 衍生指标最新快照 (最近 5 个月)")
    print("─" * 72)

    derived_cols = [c for c in df.columns if c not in ("Date", "CN10Y", "US10Y", "CPI", "PMI")]
    if "YM" in derived_cols:
        derived_cols.remove("YM")

    print(df[["Date"] + derived_cols].tail(5).to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────
# 4. 接入 Cycle6 六周期模型
# ─────────────────────────────────────────────────────────────────────

def run_cycle_model(df_raw: pd.DataFrame):
    """
    将 Wind EDB 原始数据喂入 Cycle6 六周期模型。

    Cycle6 模型通过关键字匹配列名来自动识别数据：
      - 货币因子: 需要利率相关的列名（如 "逆回购利率:7天" 等）
      - 信用因子: 需要 "中长期贷款" 相关列
      - 增长因子: 需要含 "PMI" 的列
      - M1/M2: 需要含 "M1" 和 "同比" 的列
      - 通胀因子: 需要含 "CPI" 和 "PPI" 的列

    注意：Wind EDB 采集的原始数据和 Cycle6 期望的列名不完全一致，
    需要先重命名为模型认可的格式。
    """
    from macro_cycles_CN6_excel import Cycle6

    # ── 构建 Cycle6 所需的日频 DataFrame ──

    # 日频时间轴
    daily_idx = pd.date_range(df_raw["Date"].min(), df_raw["Date"].max(), freq="D")
    daily = pd.DataFrame(index=daily_idx)

    # 将月频数据填充到日频
    macro_cols = {
        "SF_total": "社会融资规模:存量:当月值",
        "SF_ratio": "社会融资规模:存量:同比",
        "CPI": "中国:CPI:当月同比",
        "PMI": "中国:制造业PMI",
        "CN10Y": "中国:国债收益率:10年",
        "US10Y": "美国:国债收益率:10年",
    }

    for col, new_name in macro_cols.items():
        if col in df_raw.columns:
            # 月频 → 日频 (前向填充)
            monthly = df_raw.set_index("Date")[col]
            daily[new_name] = monthly.reindex(daily.index, method="ffill")

    # M1/M2 特殊处理 (原 Cycle6 期望 "中国:M1:同比" / "中国:M2:同比")
    for wind_col, cycle_col in [
        ("M1_YoY", "中国:M1:同比"),
        ("M2_YoY", "中国:M2:同比"),
    ]:
        if wind_col in df_raw.columns:
            monthly = df_raw.set_index("Date")[wind_col]
            daily[cycle_col] = monthly.reindex(daily.index, method="ffill")

    # ── 添加 M1-M2 剪刀差衍生列 ──
    if "M1-YoY" in daily.columns and "M2-YoY" in daily.columns:
        daily["M1-M2"] = daily["中国:M1:同比"] - daily["中国:M2:同比"]

    print(f"\n  Daily DataFrame: {daily.shape[0]} 天 × {daily.shape[1]} 列")
    print(f"  有效列: {list(daily.columns)}")

    # ── 实例化 Cycle6 ──
    m = Cycle6(since=str(df_raw["Date"].min().year) + "0101", use_latest_credit=False)

    # ── 计算因子 ──
    print("\n  🔄 计算宏观因子 ...")
    df_factors = m._calc(daily)
    print(f"  因子 DataFrame: {df_factors.shape}")
    print(df_factors.tail(8).to_string())

    # ── 计算周期信号 ──
    print("\n  🔄 计算六周期划分 ...")
    cycles = m.calc_cycle(df_factors, align="naive", returns="span")
    print(f"  周期区间段数: {len(cycles)}")
    print()

    # ── 行级周期状态 ──
    status = m.gen_status(cycles, freq="D")
    print(f"  日频状态序列: {len(status)} 天")
    print()

    return cycles, status, df_factors


def print_cycle_results(cycles: pd.DataFrame, status: pd.Series):
    """格式化输出周期划分结果"""
    print("=" * 72)
    print("  🔄 六周期划分结果")
    print("=" * 72)

    if cycles is None or cycles.empty:
        print("  (无有效周期划分)")
        return

    # ── 周期区间表 ──
    print("\n  ┌───── 周期区间表 ─────────────────────────────────┐")
    print(f"  │  {'区间':^18s}  │ {'周期':>6s} │ {'名称':<10s} │")
    print("  ├──────────────────────────────────────────────────┤")
    for _, row in cycles.iterrows():
        c = int(row["c"])
        name = CYCLE_MAP.get(c, "未知")
        color = CYCLE_COLORS.get(c, "⬜")
        since = row["since"].strftime("%Y-%m") if hasattr(row["since"], "strftime") else str(row["since"])[:7]
        till = row["till"].strftime("%Y-%m") if hasattr(row["till"], "strftime") else str(row["till"])[:7]
        duration = (row["till"] - row["since"]).days if hasattr(row["till"], "__sub__") else 0
        print(f"  │  {since} ~ {till}  │ {color} {c:>2d}  │ {name:<10s} │ ({duration:>3d}天)")
    print("  └──────────────────────────────────────────────────┘")

    # ── 周期持续时间统计 ──
    print("\n  ┌───── 周期持续时间统计 ─────────────────────────────┐")
    cycles_copy = cycles.copy()
    cycles_copy["duration_days"] = (cycles_copy["till"] - cycles_copy["since"]).dt.days
    stats = cycles_copy.groupby("c").agg(
        出现次数=("duration_days", "count"),
        总天数=("duration_days", "sum"),
        平均天数=("duration_days", "mean"),
    ).round(1)
    stats.index = stats.index.map(lambda x: f"{CYCLE_MAP.get(int(x), '未知')} ({int(x)})")
    print(stats.to_string())
    print("  └──────────────────────────────────────────────────┘")

    # ── 当前周期 ──
    if status is not None and len(status) > 0:
        latest_date = status.index[-1]
        latest_cycle = int(status.iloc[-1])
        print(f"\n  🎯 当前周期 ({latest_date:%Y-%m-%d}): {CYCLE_COLORS.get(latest_cycle, '')} "
              f"{CYCLE_MAP.get(latest_cycle, '未知')} (周期 {latest_cycle})")


# ─────────────────────────────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   宏观因子 Wind EDB → DataFrame → 六周期划分           ║")
    print("║   端到端案例演示                                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  🕐 运行时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print(f"  📁 项目路径: {SCRIPT_DIR}")
    print()

    # ── Step 0: 展示配置 ──
    show_macro_mappings()

    # ── Step 1: 从 Wind EDB 提取数据 ──
    if _wind_available():
        print("  ✅ WindPy 已连接，从 Wind EDB 拉取实时数据\n")
        df_raw = fetch_all_from_wind()
    else:
        print("  ⚠️  WindPy 未连接 / 不可用")
        print("  ── 进入演示模式：展示预期数据结构和处理流程 ──\n")
        print("  💡 启动 Wind 终端后，本脚本将从 EDB 拉取:")
        for name, info in MACRO_MAPPINGS.items():
            print(f"     {name:12s} (ticker={info['ticker']})")
        print()
        print("  以下展示处理流程的伪代码:")
        print()
        # 生成一个模拟的 DataFrame 用于展示结构
        dates = pd.date_range("2023-01-01", "2026-06-01", freq="ME")
        np.random.seed(42)
        df_demo = pd.DataFrame({
            "Date": dates,
            "SF_total": np.linspace(350, 420, len(dates)),
            "SF_ratio": 8 + 3 * np.sin(np.linspace(0, 2 * np.pi, len(dates))),
            "M1_YoY": 2 + 1.5 * np.cos(np.linspace(0, 2 * np.pi, len(dates))) + np.random.randn(len(dates)) * 0.5,
            "M2_YoY": 8 + 2 * np.cos(np.linspace(0, 1.5 * np.pi, len(dates))) + np.random.randn(len(dates)) * 0.3,
            "CPI": 1.5 + 0.8 * np.sin(np.linspace(0, 2 * np.pi, len(dates))),
            "PMI": 50 + 2 * np.sin(np.linspace(0, 1.5 * np.pi, len(dates))),
            "CN10Y": 2.5 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, len(dates))),
            "US10Y": 4.0 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, len(dates))),
        })
        df_raw = df_demo
        print_dataframe_snapshot(df_raw, "模拟宏观因子数据 (演示模式)")
        print("  ⚠️  以上为模拟数据，实际运行时将替换为 Wind EDB 真实数据")
        print()

    # ── Step 2: 展示原始 DataFrame ──
    print_dataframe_snapshot(df_raw, f"从 Wind EDB 提取的宏观因子原始数据")

    # ── Step 3: 计算衍生指标 ──
    print("─" * 72)
    print("  衍生指标计算")
    print("─" * 72)
    df_derived = compute_derived_indicators(df_raw)
    print_derived_snapshot(df_derived)

    # ── Step 4: 接入 Cycle6 模型 ──
    if _wind_available():
        print("─" * 72)
        print("  六周期模型运算")
        print("─" * 72)
        try:
            cycles, status, factors = run_cycle_model(df_raw)
            print_cycle_results(cycles, status)
        except Exception as e:
            print(f"\n  ⚠️  周期模型运算失败: {e}")
            print("  (可能缺少某些必需列，但数据提取和 DataFrame 构建已完成)")
            import traceback
            traceback.print_exc()
    else:
        print()
        print("─" * 72)
        print("  📌 接入 Cycle6 模型 (演示模式)")
        print("─" * 72)
        print()
        print("  启动 Wind 终端后，本脚本将执行以下完整流程:")
        print()
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Wind EDB               月频原始数据  日频重采样        │")
        print("  │  ────────              ───────────  ──────────         │")
        print("  │  M0000551(M1同比  ) ──→  M1_YoY   ──→  中国:M1:同比    │")
        print("  │  M0001385(M2同比  ) ──→  M2_YoY   ──→  中国:M2:同比    │")
        print("  │  M5525546(社融存量) ──→  SF_total ──→  (社融脉冲)      │")
        print("  │  M5525547(社融同比) ──→  SF_ratio ──→  信用因子         │")
        print("  │  M0000612(CPI同比 ) ──→  CPI      ──→  通胀因子         │")
        print("  │  M0017126(PMI     ) ──→  PMI      ──→  增长因子         │")
        print("  │  M0325687(中国10Y ) ──→  CN10Y    ──→  货币因子         │")
        print("  │  G0000891(美国10Y ) ──→  US10Y    ──→  (跨境参考)       │")
        print("  │                              ↓                         │")
        print("  │                  Cycle6.calc_cycle()                   │")
        print("  │                              ↓                         │")
        print("  │                  六周期划分结果                        │")
        print("  │     信用扩张 | 复苏 | 货币退潮 | 信用退潮              │")
        print("  │     经济放缓 | 货币扩张                                │")
        print("  └────────────────────────────────────────────────────────┘")
        print()

    print("\n" + "=" * 72)
    print("  ✅ 案例演示完成")
    print("=" * 72)
    print()
    print("  关键发现:")
    print(f"    - 新增 M1/M2 同比映射 (M0000551, M0001385)")
    print(f"    - WindDataLoader 已增加 _fetch_m1m2() 方法")
    print(f"    - data_hub_config.json macro_mappings 已扩展为 8 个指标")
    print(f"    - 衍生指标: M1-M2 剪刀差 + 社融脉冲 + 信用因子")
    print()
    print("  后续可以:")
    print("    1. 启动 Wind → python demo_macro_wind_to_cycle.py 直接运行")
    print("    2. 将新增指标纳入 update_allCycleInput_wind.py 的宏观工作表")
    print("    3. 研究 M1-M2 剪刀差在不同周期状态下的预测能力")


if __name__ == "__main__":
    main()
