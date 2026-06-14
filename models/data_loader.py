"""
数据加载器 — Data Loader Module

统一的数据获取抽象层，支持从 Excel 文件和 Wind API 两种数据源
获取资产日频收益率和月频宏观因子数据。

设计原则：
- 接口统一：BaseDataLoader → ExcelDataLoader / WindDataLoader
- 配置驱动：asset_mappings / macro_mappings 从 data_hub_config.json 读取
- 增量获取：支持 start_date 只拉取增量数据
- 容错机制：Wind API 调用含自动重试
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===================================================================
# 抽象基类 | Abstract Base Data Loader
# ===================================================================

class BaseDataLoader(ABC):
    """
    数据加载器抽象基类。

    Parameters
    ----------
    config : dict
        配置字典，包含 data_source, excel_path, asset_mappings, macro_mappings 等。
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load_returns(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载资产的日频收益率数据。

        Parameters
        ----------
        assets : List[str]
            需要获取的资产名称列表。
        start_date : str or None
            起始日期 "YYYY-MM-DD"，None 表示从最早数据开始。
        end_date : str or None
            截止日期，None 表示最新。

        Returns
        -------
        pd.DataFrame
            列: ['Date', asset_1, asset_2, ...]
        """
        ...

    @abstractmethod
    def load_macro_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载月频宏观因子数据。

        Parameters
        ----------
        start_date : str or None
            起始日期。
        end_date : str or None
            截止日期。

        Returns
        -------
        pd.DataFrame
            列: ['Date', 'CPI', 'PMI', 'CN10Y', 'US10Y', 'SF_ratio', 'SF_total', ...]
        """
        ...


# ===================================================================
# Excel 数据加载器 | Excel Data Loader
# ===================================================================

class ExcelDataLoader(BaseDataLoader):
    """
    从 Excel 文件读取收益率和宏观因子数据。

    数据源约定位于 config["excel_path"]，包含 'return' 和 'macroData' 两个工作表。
    """

    def _resolve_path(self, config_path: str) -> str:
        """
        将配置文件中的相对路径解析为绝对路径。

        相对于项目根目录（包含 models/ 的上一级）。
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, config_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"未找到 Excel 数据源: {full_path}")
        return full_path

    def _read_sheet(self, sheet_name: str) -> pd.DataFrame:
        """统一读取指定工作表"""
        excel_path = self.config.get("excel_path", "data/input/allCycleInput.xlsx")
        full_path = self._resolve_path(excel_path)
        return pd.read_excel(full_path, sheet_name=sheet_name)

    def load_returns(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._read_sheet("return")
        df["Date"] = pd.to_datetime(df["Date"])

        cols = ["Date"] + [a for a in assets if a in df.columns]
        df = df[cols]

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)

    def load_macro_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._read_sheet("macroData")
        df["Date"] = pd.to_datetime(df["Date"])

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)


# ===================================================================
# Wind API 数据加载器 | Wind Data Loader
# ===================================================================

class WindDataLoader(BaseDataLoader):
    """
    通过 Wind API (WindPy) 实时获取资产收益率和宏观因子数据。

    Wind 接口调用含自动重试机制（最多 RETRY_MAX 次），
    应对 Wind 终端偶发性接口超时或连接断开。

    Parameters
    ----------
    config : dict
        必需键: asset_mappings, macro_mappings
        可选键: wind_retry_max (默认 3), wind_retry_delay (默认 2 秒)
    """

    RETRY_MAX: int = 3
    RETRY_DELAY: float = 2.0

    def __init__(self, config: dict):
        super().__init__(config)
        self._wind_started = False

    # ── Wind 连接管理 ──

    def _start_wind(self) -> None:
        """初始化并验证 Wind API 连接"""
        if self._wind_started:
            return

        try:
            from WindPy import w
        except ImportError:
            raise ImportError(
                "未检测到 WindPy 库，请确保已安装 Wind 终端及其 Python API 接口。"
            )

        w.start()
        if not w.isconnected():
            raise ConnectionError(
                "无法连接至 Wind API，请确认 Wind 终端已启动并登录！"
            )
        self._wind_started = True

    def _call_with_retry(self, func, *args, **kwargs):
        """
        带自动重试的 Wind API 调用封装。

        当 ErrorCode 非零时自动重试，最多 RETRY_MAX 次。
        """
        last_error = None
        for attempt in range(1, self.RETRY_MAX + 1):
            result = func(*args, **kwargs)
            if result.ErrorCode == 0:
                return result

            last_error = (result.ErrorCode, result.Data)
            if attempt < self.RETRY_MAX:
                time.sleep(self.RETRY_DELAY)

        raise RuntimeError(
            f"Wind API 调用失败 (已重试 {self.RETRY_MAX} 次), "
            f"ErrorCode={last_error[0]}, Data={last_error[1]}"
        )

    # ── 收益率数据 ──

    def _parse_asset_mappings(self, assets: List[str]) -> Tuple[List[str], List[str], dict]:
        """
        根据配置的 asset_mappings，将资产名拆分为 ETF 列表和指数列表。

        Returns
        -------
        etf_tickers : List[str]
            需用 wsd 提取复权净值的 ETF。
        idx_tickers : List[str]
            需用 wsd 提取收盘价的指数。
        asset_to_ticker : dict
            {asset_name: (ticker, type)}
        """
        mappings = self.config.get("asset_mappings", {})
        etf_tickers: List[str] = []
        idx_tickers: List[str] = []
        asset_to_ticker: dict = {}

        for asset in assets:
            if asset not in mappings:
                raise ValueError(
                    f"资产 '{asset}' 的 Wind 映射未定义，"
                    f"请在配置文件的 asset_mappings 中添加。"
                )
            ticker = mappings[asset]["ticker"]
            atype = mappings[asset]["type"]
            asset_to_ticker[asset] = (ticker, atype)
            if atype == "ETF":
                etf_tickers.append(ticker)
            else:
                idx_tickers.append(ticker)

        return etf_tickers, idx_tickers, asset_to_ticker

    def _rename_ticker_columns(
        self, df: pd.DataFrame, asset_to_ticker: dict, assets: List[str]
    ) -> pd.DataFrame:
        """将 Wind 返回的 ticker 列名重命名为资产中文名"""
        final_cols = ["Date"]
        for asset in assets:
            ticker, _ = asset_to_ticker[asset]
            matched = None
            for col in df.columns:
                if str(col).upper() == str(ticker).upper():
                    matched = col
                    break
            if matched is not None:
                df = df.rename(columns={matched: asset})
            else:
                df[asset] = 0.0
            final_cols.append(asset)
        return df[final_cols]

    def load_returns(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        self._start_wind()
        from WindPy import w

        start = start_date or "2011-01-01"
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")

        etf_tickers, idx_tickers, asset_to_ticker = self._parse_asset_mappings(assets)

        # 获取 ETF 复权净值
        df_etf = pd.DataFrame()
        if etf_tickers:
            res_etf = self._call_with_retry(
                w.wsd, ",".join(etf_tickers), "nav_adj", start, end, ""
            )
            df_etf = pd.DataFrame(
                res_etf.Data,
                index=res_etf.Codes,
                columns=pd.to_datetime(res_etf.Times),
            ).T

        # 获取指数收盘价
        df_idx = pd.DataFrame()
        if idx_tickers:
            res_idx = self._call_with_retry(
                w.wsd, ",".join(idx_tickers), "close", start, end, ""
            )
            df_idx = pd.DataFrame(
                res_idx.Data,
                index=res_idx.Codes,
                columns=pd.to_datetime(res_idx.Times),
            ).T

        # 合并并计算收益率
        df_combined = pd.concat([df_etf, df_idx], axis=1)
        df_combined = df_combined.sort_index().ffill()
        df_rets = df_combined.pct_change().dropna(how="all")

        # 现金资产特殊处理: 日收益 = 前日收盘价 / 100 / 365
        if "M0220163" in df_combined.columns:
            df_rets["M0220163"] = df_combined["M0220163"].shift(1) / 100 / 365
            df_rets["M0220163"] = df_rets["M0220163"].fillna(0)

        df_rets = df_rets.reset_index().rename(columns={"index": "Date"})
        return self._rename_ticker_columns(df_rets, asset_to_ticker, assets).reset_index(drop=True)

    # ── 宏观因子数据 ──

    def _fetch_yields(self, macro_mappings: dict, start: str, end: str):
        """拉取 CN10Y 和 US10Y 日度收益率"""
        from WindPy import w

        cn10y = macro_mappings.get("CN10Y", {}).get("ticker", "M0325687")
        us10y = macro_mappings.get("US10Y", {}).get("ticker", "G0000891")

        res = self._call_with_retry(w.edb, [cn10y, us10y], start, end)
        df = pd.DataFrame(
            res.Data,
            index=["CN10Y", "US10Y"],
            columns=pd.to_datetime(res.Times),
        ).T
        df.index.name = "Date"
        df = df.reset_index()
        df[["CN10Y", "US10Y"]] = df[["CN10Y", "US10Y"]].ffill().bfill()
        return df

    def _fetch_monthly_macro(self, macro_mappings: dict, start: str, end: str):
        """拉取 CPI 和 PMI 月频数据"""
        from WindPy import w

        cpi = macro_mappings.get("CPI", {}).get("ticker", "M0000612")
        pmi = macro_mappings.get("PMI", {}).get("ticker", "M0017126")

        res = self._call_with_retry(w.edb, [cpi, pmi], start, end)
        df = pd.DataFrame(
            res.Data,
            index=["CPI", "PMI"],
            columns=pd.to_datetime(res.Times),
        ).T
        df.index.name = "Date"
        df = df.reset_index()
        df["YM"] = df["Date"].dt.to_period("M")
        return df.groupby("YM")[["CPI", "PMI"]].last()

    def _fetch_social_financing(self, macro_mappings: dict, start: str, end: str):
        """拉取社融数据（SF_ratio, SF_total），可选"""
        from WindPy import w

        sf_ratio = macro_mappings.get("SF_ratio", {}).get("ticker")
        sf_total = macro_mappings.get("SF_total", {}).get("ticker")
        if not sf_ratio or not sf_total:
            return None

        try:
            res = self._call_with_retry(w.edb, [sf_ratio, sf_total], start, end)
            df = pd.DataFrame(
                res.Data,
                index=["SF_ratio", "SF_total"],
                columns=pd.to_datetime(res.Times),
            ).T
            df.index.name = "Date"
            df = df.reset_index()
            df["YM"] = df["Date"].dt.to_period("M")
            return df.groupby("YM")[["SF_ratio", "SF_total"]].last()
        except RuntimeError:
            return None

    def _fetch_m1m2(self, macro_mappings: dict, start: str, end: str):
        """
        拉取 M1 / M2 同比增速，月频。

        Wind EDB 代码:
            M0000551 — M1:同比
            M0001385 — M2:同比
        """
        from WindPy import w

        m1 = macro_mappings.get("M1_YoY", {}).get("ticker")
        m2 = macro_mappings.get("M2_YoY", {}).get("ticker")
        if not m1 or not m2:
            return None

        try:
            res = self._call_with_retry(w.edb, [m1, m2], start, end)
            df = pd.DataFrame(
                res.Data,
                index=["M1_YoY", "M2_YoY"],
                columns=pd.to_datetime(res.Times),
            ).T
            df.index.name = "Date"
            df = df.reset_index()
            df["YM"] = df["Date"].dt.to_period("M")
            return df.groupby("YM")[["M1_YoY", "M2_YoY"]].last()
        except RuntimeError:
            return None

    def load_macro_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        self._start_wind()

        start = start_date or "2011-01-01"
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        macro_mappings = self.config.get("macro_mappings", {})

        # 1. 日度收益率（聚合成月均值）
        df_yields = self._fetch_yields(macro_mappings, start, end)
        df_yields["YM"] = pd.to_datetime(df_yields["Date"]).dt.to_period("M")
        monthly_yield_avg = df_yields.groupby("YM")[["CN10Y", "US10Y"]].mean()

        # 2. 月度 CPI / PMI
        df_macro_wind = self._fetch_monthly_macro(macro_mappings, start, end)

        # 3. 社融（可选）
        df_sf = self._fetch_social_financing(macro_mappings, start, end)

        # 4. M1 / M2 同比（可选）
        df_m1m2 = self._fetch_m1m2(macro_mappings, start, end)

        # 5. 统一月度时间索引
        monthly_range = pd.date_range(
            start=pd.to_datetime(start), end=pd.to_datetime(end), freq="ME"
        )
        df_macro = pd.DataFrame(index=monthly_range)
        df_macro.index.name = "Date"
        df_macro = df_macro.reset_index()
        df_macro["YM"] = df_macro["Date"].dt.to_period("M")
        df_macro = df_macro.set_index("YM")

        for col in ["CN10Y", "US10Y", "CPI", "PMI", "SF_ratio", "SF_total", "M1_YoY", "M2_YoY"]:
            df_macro[col] = np.nan

        df_macro.update(monthly_yield_avg)
        df_macro.update(df_macro_wind)
        if df_sf is not None:
            df_macro.update(df_sf)
        if df_m1m2 is not None:
            df_macro.update(df_m1m2)

        df_macro = df_macro.reset_index(drop=True)
        df_macro = df_macro.ffill().bfill()
        return df_macro
