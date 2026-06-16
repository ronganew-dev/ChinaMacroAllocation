# -*- coding: utf-8 -*-
"""
Module: data_hub.py
Description: 数据中台 — 简化版

基于 WindFetcher 的薄封装，提供:
  - get_returns(): 日频资产收益率
  - get_macro():   月频宏观因子
  - get_universe(): 资产池管理

配置来源: config/asset_tickers.csv + config/macro_tickers.csv
缓存位置: data/cache/*.parquet
"""

import os
from typing import Optional

import pandas as pd

from models.data_loader import WindFetcher
from models.universe import Universe


class DataHub:
    """
    数据中台 — 统一数据入口。

    Parameters
    ----------
    cache_dir : str, optional
        Parquet 缓存目录
    start_date : str, optional
        数据起始日期
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        start_date: str = "2011-01-01",
    ):
        self.fetcher = WindFetcher(cache_dir=cache_dir, start_date=start_date)
        self.universe = Universe()
        self._cached_returns: Optional[pd.DataFrame] = None
        self._cached_macro: Optional[pd.DataFrame] = None

    def get_universe(self) -> Universe:
        return self.universe

    def get_safe_asset(self) -> str:
        return self.universe.get_safe_asset()

    def get_returns(
        self,
        assets: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取日频资产收益率。

        Parameters
        ----------
        assets : list of str, optional
            资产名称列表，None 表示全部
        start_date, end_date : str, optional
            时间区间
        use_cache : bool
            是否使用 Parquet 缓存
        """
        if self._cached_returns is None or not use_cache:
            self._cached_returns = self.fetcher.fetch_returns(
                asset_names=assets, start_date=start_date,
                end_date=end_date, use_cache=use_cache,
            )
        df = self._cached_returns.copy()
        if assets:
            cols = ["Date"] + [a for a in assets if a in df.columns]
            df = df[cols]
        return df

    def get_macro(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取月频宏观因子数据。
        """
        if self._cached_macro is None or not use_cache:
            self._cached_macro = self.fetcher.fetch_macro(
                start_date=start_date, end_date=end_date,
                use_cache=use_cache,
            )
        return self._cached_macro.copy()
