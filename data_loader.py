# -*- coding: utf-8 -*-
"""
data_loader — 向后兼容桥接层

核心实现已迁移至 models/data_loader.py。
本文件仅做重导出，保持向后兼容。

新代码请直接使用:
    from models.data_loader import WindFetcher
"""

from models.data_loader import WindFetcher, read_asset_config, read_macro_config

__all__ = [
    "WindFetcher",
    "read_asset_config",
    "read_macro_config",
]
