# -*- coding: utf-8 -*-
"""
Module: models.data_loader
Description: 数据加载与缓存模块 — CSV驱动 + Parquet缓存

架构
----
  CSV配置 (config/*.csv)  →  WindFetcher  →  Parquet缓存 (data/cache/*.parquet)
                                      ↘  DataFrame

核心理念
--------
  1. CSV 配置：手工增删行即可新增/删除数据源，无需改代码
  2. Parquet 缓存：Wind 取数后自动存盘，下次优先读本地
  3. 扁平设计：一个函数取一个序列，不搞复杂继承

用法
----
  from models.data_loader import WindFetcher

  fetcher = WindFetcher()
  df_macro = fetcher.fetch_macro()          # 读取/更新宏观数据
  df_ret   = fetcher.fetch_returns()        # 读取/更新资产收益率
  df_raw   = fetcher.fetch_series("CPI")   # 取单个序列
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 路径常量
# ═══════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

ASSET_CSV = _CONFIG_DIR / "asset_tickers.csv"
MACRO_CSV = _CONFIG_DIR / "macro_tickers.csv"


# ═══════════════════════════════════════════════════════════════
# CSV 配置读取
# ═══════════════════════════════════════════════════════════════

def read_asset_config(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    读取资产配置 CSV。

    Returns
    -------
    pd.DataFrame
        列: name, ticker, type, freq, field, desc
    """
    path = Path(csv_path) if csv_path else ASSET_CSV
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    required = {"name", "ticker", "type", "freq", "field"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"资产配置 CSV 缺少必需列: {missing}")
    return df


def read_macro_config(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    读取宏观指标配置 CSV。

    Returns
    -------
    pd.DataFrame
        列: name, ticker, freq, group, desc
    """
    path = Path(csv_path) if csv_path else MACRO_CSV
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    required = {"name", "ticker", "freq", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"宏观配置 CSV 缺少必需列: {missing}")
    return df


# ═══════════════════════════════════════════════════════════════
# WindFetcher — 核心取数类
# ═══════════════════════════════════════════════════════════════

class WindFetcher:
    """
    从 Wind EDB / wsd 获取数据，自动缓存为 Parquet。

    流程
    ----
      1. 读 CSV 配置 → 知道要取哪些序列
      2. 检查 Parquet 缓存 → 有则增量更新，无则全量拉取
      3. 调 Wind API → 存 Parquet → 返回 DataFrame

    Parameters
    ----------
    cache_dir : str or Path, optional
        Parquet 缓存目录，默认 data/cache/
    start_date : str, optional
        数据起始日期，默认 "2011-01-01"
    retry_max : int
        Wind API 调用失败重试次数，默认 3
    retry_delay : float
        重试间隔秒数，默认 2.0
    """

    RETRY_MAX: int = 3
    RETRY_DELAY: float = 2.0

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        start_date: str = "2011-01-01",
        retry_max: int = 3,
        retry_delay: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.retry_max = retry_max
        self.retry_delay = retry_delay
        self._wind_started = False

        self.asset_cfg = read_asset_config()
        self.macro_cfg = read_macro_config()

    # ── Wind 连接 ──

    def _ensure_wind(self) -> None:
        """初始化 Wind 连接"""
        if self._wind_started:
            return
        from WindPy import w
        w.start()
        if not w.isconnected():
            raise ConnectionError("Wind 终端未连接，请确认已启动并登录")
        self._wind_started = True

    def _call_with_retry(self, func, *args, **kwargs):
        """带自动重试的 Wind API 调用"""
        last_error = None
        for attempt in range(1, self.retry_max + 1):
            result = func(*args, **kwargs)
            if result.ErrorCode == 0:
                return result
            last_error = (result.ErrorCode, result.Data)
            if attempt < self.retry_max:
                time.sleep(self.retry_delay)
        raise RuntimeError(
            f"Wind API 调用失败 (重试 {self.retry_max} 次), "
            f"ErrorCode={last_error[0]}"
        )

    # ── 单序列取数 ──

    def fetch_series(
        self,
        name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取单个数据序列。

        优先读 Parquet 缓存，缓存不存在或 use_cache=False 时调 Wind。

        Parameters
        ----------
        name : str
            序列名称（必须在 CSV 配置中存在）
        start_date : str, optional
            起始日期
        end_date : str, optional
            截止日期
        use_cache : bool
            是否优先使用 Parquet 缓存

        Returns
        -------
        pd.DataFrame
            列: [Date, <name>]
        """
        start = start_date or self.start_date
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"{name}.parquet"

        # 1. 尝试读缓存
        if use_cache and cache_path.exists():
            df_cached = pd.read_parquet(cache_path)
            df_cached["Date"] = pd.to_datetime(df_cached["Date"])
            cached_end = df_cached["Date"].max().strftime("%Y-%m-%d")
            if cached_end >= end:
                df = df_cached[df_cached["Date"] >= pd.to_datetime(start)].copy()
                return df.reset_index(drop=True)
            # 增量更新
            start = pd.Timestamp(cached_end).strftime("%Y-%m-%d")
            df_new = self._fetch_from_wind(name, start, end)
            if df_new is not None and len(df_new) > 0:
                df_new = df_new[df_new["Date"] > df_cached["Date"].max()]
                df_all = pd.concat([df_cached, df_new], ignore_index=True)
                df_all.to_parquet(cache_path, index=False)
                return df_all[df_all["Date"] >= pd.to_datetime(start_date or self.start_date)].reset_index(drop=True)
            return df_cached[df_cached["Date"] >= pd.to_datetime(start_date or self.start_date)].reset_index(drop=True)

        # 2. 无缓存，全量拉取
        df = self._fetch_from_wind(name, start, end)
        if df is not None and len(df) > 0:
            df.to_parquet(cache_path, index=False)
        return df if df is not None else pd.DataFrame(columns=["Date", name])

    def _fetch_from_wind(
        self, name: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """
        从 Wind API 拉取单个序列的原始数据。

        自动识别资产/宏观配置，选择 wsd 或 edb 接口。
        """
        self._ensure_wind()
        from WindPy import w

        # 先查资产配置
        asset_row = self.asset_cfg[self.asset_cfg["name"] == name]
        if len(asset_row) > 0:
            row = asset_row.iloc[0]
            ticker = row["ticker"]
            field = row["field"]
            res = self._call_with_retry(w.wsd, ticker, field, start, end, "")
            if res.ErrorCode == 0 and res.Data and len(res.Data[0]) > 0:
                df = pd.DataFrame({
                    "Date": pd.to_datetime(res.Times),
                    name: res.Data[0],
                })
                return df

        # 再查宏观配置
        macro_row = self.macro_cfg[self.macro_cfg["name"] == name]
        if len(macro_row) > 0:
            row = macro_row.iloc[0]
            ticker = row["ticker"]
            res = self._call_with_retry(w.edb, ticker, start, end)
            if res.ErrorCode == 0 and res.Data and len(res.Data[0]) > 0:
                df = pd.DataFrame({
                    "Date": pd.to_datetime(res.Times),
                    name: res.Data[0],
                })
                return df

        print(f"  ⚠️ '{name}' 未在 CSV 配置中找到，跳过")
        return None

    # ── 批量取数 ──

    def fetch_macro(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        批量获取所有宏观指标，合并为月频 DataFrame。

        日频指标 → 月均值聚合
        月频指标 → 取每月最后一个值

        Returns
        -------
        pd.DataFrame
            列: [Date, <macro_name_1>, <macro_name_2>, ...]
        """
        start = start_date or self.start_date
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        series_list = []

        for _, row in self.macro_cfg.iterrows():
            name = row["name"]
            freq = row["freq"]
            print(f"  📡 {name} ({row['ticker']}, {freq}频) ...", end=" ")

            try:
                df_s = self.fetch_series(name, start, end, use_cache=use_cache)
            except Exception as e:
                print(f"⚠️ 失败 ({e})")
                continue
            if df_s is None or len(df_s) == 0 or name not in df_s.columns:
                print("⚠️ 无数据")
                continue

            # 按频率聚合
            df_s["Date"] = pd.to_datetime(df_s["Date"])
            df_s = df_s.dropna(subset=[name])
            if len(df_s) == 0:
                print("⚠️ 全 NaN")
                continue

            df_s["YM"] = df_s["Date"].dt.to_period("M")
            if freq == "D":
                monthly = df_s.groupby("YM")[name].mean()
            else:
                monthly = df_s.groupby("YM")[name].last()

            monthly = monthly.reset_index()
            monthly["Date"] = monthly["YM"].dt.to_timestamp(how="end").dt.normalize()
            monthly = monthly[["Date", name]]
            series_list.append(monthly)
            print(f"✅ {len(monthly)} 月")

        if not series_list:
            return pd.DataFrame()

        # 以日期为 key 外连接
        df = series_list[0]
        for s in series_list[1:]:
            df = pd.merge(df, s, on="Date", how="outer")
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def fetch_returns(
        self,
        asset_names: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        批量获取资产日频收益率。

        ETF → 取复权净值 → 算日收益率
        Index → 取收盘价 → 算日收益率
        特殊处理: 现金资产 (M0220163) → 日收益 = 前日收盘价 / 100 / 365

        Parameters
        ----------
        asset_names : list of str, optional
            资产名称列表，None 表示取全部

        Returns
        -------
        pd.DataFrame
            列: [Date, <asset_name_1>, <asset_name_2>, ...]
        """
        start = start_date or self.start_date
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        cfg = self.asset_cfg
        if asset_names:
            cfg = cfg[cfg["name"].isin(asset_names)]

        price_series = []
        cash_assets = cfg[cfg["ticker"] == "M0220163"]["name"].tolist()

        for _, row in cfg.iterrows():
            name = row["name"]
            print(f"  📡 {name} ({row['ticker']}) ...", end=" ")

            try:
                df_s = self.fetch_series(name, start, end, use_cache=use_cache)
            except Exception as e:
                print(f"⚠️ 失败 ({e})")
                continue
            if df_s is None or len(df_s) == 0 or name not in df_s.columns:
                print("⚠️ 无数据")
                continue

            df_s["Date"] = pd.to_datetime(df_s["Date"])
            # 存价格，最后统一算收益率
            price_series.append(df_s[["Date", name]].rename(columns={name: f"_price_{name}"}))
            print(f"✅ {len(df_s)} 日")

        if not price_series:
            return pd.DataFrame()

        # 合并所有价格序列
        df_price = price_series[0]
        for s in price_series[1:]:
            df_price = pd.merge(df_price, s, on="Date", how="outer")
        df_price = df_price.sort_values("Date").ffill()

        # 算收益率
        result = df_price[["Date"]].copy()
        for col in df_price.columns:
            if col.startswith("_price_"):
                asset_name = col.replace("_price_", "")
                if asset_name in cash_assets:
                    # 现金: 日收益 = 前日值 / 100 / 365
                    result[asset_name] = df_price[col].shift(1) / 100 / 365
                    result[asset_name] = result[asset_name].fillna(0)
                else:
                    result[asset_name] = df_price[col].pct_change()

        result = result.dropna(how="all", subset=[c for c in result.columns if c != "Date"])
        return result.reset_index(drop=True)

    # ── 缓存管理 ──

    def list_cache(self) -> pd.DataFrame:
        """列出所有 Parquet 缓存文件及其行数和日期范围"""
        rows = []
        for p in sorted(self.cache_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            date_cols = [c for c in df.columns if "date" in c.lower()]
            if date_cols:
                dates = pd.to_datetime(df[date_cols[0]])
                date_range = f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}"
            else:
                date_range = "N/A"
            rows.append({
                "name": p.stem,
                "rows": len(df),
                "cols": list(df.columns),
                "date_range": date_range,
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        return pd.DataFrame(rows)

    def clear_cache(self, names: Optional[list] = None) -> int:
        """
        清除 Parquet 缓存。

        Parameters
        ----------
        names : list of str, optional
            要清除的序列名，None 表示清除全部

        Returns
        -------
        int
            删除的文件数
        """
        count = 0
        for p in sorted(self.cache_dir.glob("*.parquet")):
            if names is None or p.stem in names:
                p.unlink()
                count += 1
        return count
