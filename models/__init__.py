"""
models — 数据模型与加载器

统一的底层数据抽象层，为上层策略提供标准化的数据接口。

导出
----
BaseDataLoader
    抽象基类，定义 load_returns / load_macro_data 接口契约。
ExcelDataLoader
    从 Excel 文件中读取预先准备好的收益率和宏观因子数据。
WindDataLoader
    通过 Wind API (WindPy) 实时获取数据，含自动重试机制。
Universe
    资产池管理器，支持动态筛选（排除列表 / 历史数据阈值）。
"""

from models.data_loader import BaseDataLoader, ExcelDataLoader, WindDataLoader
from models.universe import Universe

__all__ = [
    "BaseDataLoader",
    "ExcelDataLoader",
    "WindDataLoader",
    "Universe",
]
