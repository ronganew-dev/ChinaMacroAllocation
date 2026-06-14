import json
import os
import pandas as pd
from models.data_loader import ExcelDataLoader, WindDataLoader
from models.universe import Universe

class DataHub:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataHub, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        if self._initialized:
            return
            
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "data_hub_config.json")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.data_source = self.config.get("data_source", "excel").lower()
        
        # 初始化对应的加载器
        if self.data_source == "wind":
            self.loader = WindDataLoader(self.config)
        else:
            self.loader = ExcelDataLoader(self.config)
            
        # 实例化资产池管理器
        self.universe = Universe(self.config.get("universe", {}))
        
        # 缓存容器，规避重复读取或调用接口
        self._cached_returns = None
        self._cached_macro = None
        self._initialized = True

    def get_universe(self) -> Universe:
        return self.universe

    def get_safe_asset(self) -> str:
        return self.universe.get_safe_asset()

    def get_returns(self, assets, start_date=None, end_date=None) -> pd.DataFrame:
        """
        根据资产列表与时间区间获取日频收益率。
        """
        all_requested = set(assets)
        if self._cached_returns is not None:
            cached_cols = set(self._cached_returns.columns) - {'Date'}
        else:
            cached_cols = set()
            
        # 如果缓存中缺少某些请求的资产，则需要动态加载并拼接
        if not all_requested.issubset(cached_cols):
            missing_assets = list(all_requested - cached_cols)
            new_df = self.loader.load_returns(missing_assets)
            if self._cached_returns is None:
                self._cached_returns = new_df
            else:
                self._cached_returns = pd.merge(self._cached_returns, new_df, on="Date", how="outer").sort_values("Date")
                
        df = self._cached_returns.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 保证返回的列仅包含 Date 和请求的资产列（且按照请求顺序排列）
        cols = ['Date'] + [a for a in assets if a in df.columns]
        df = df[cols]
        
        # 时间过滤
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
            
        return df.reset_index(drop=True)

    def get_macro_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        加载月频宏观因子数据。
        """
        if self._cached_macro is None:
            self._cached_macro = self.loader.load_macro_data()
            
        df = self._cached_macro.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 时间过滤
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
            
        return df.reset_index(drop=True)
