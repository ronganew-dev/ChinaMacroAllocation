"""
数据加载器 — 向后兼容桥接层

核心实现已迁移至 models.data_loader。
"""

from models.data_loader import BaseDataLoader, ExcelDataLoader, WindDataLoader

__all__ = [
    "BaseDataLoader",
    "ExcelDataLoader",
    "WindDataLoader",
]
