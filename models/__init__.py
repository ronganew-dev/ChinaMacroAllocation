"""
models — 数据模型与加载器

统一的数据获取与缓存层，CSV 驱动 + Parquet 缓存。

核心类
------
WindFetcher
    CSV 配置驱动，从 Wind EDB / wsd 获取数据，自动缓存为 Parquet。
    支持单序列取数、批量宏观取数、批量资产收益率取数。

Universe
    资产池管理器，支持动态筛选。

配置文件
--------
config/asset_tickers.csv   资产 Wind 代码映射
config/macro_tickers.csv   宏观指标 Wind 代码映射
"""

from models.data_loader import WindFetcher, read_asset_config, read_macro_config
from models.universe import Universe

__all__ = [
    "WindFetcher",
    "read_asset_config",
    "read_macro_config",
    "Universe",
]
