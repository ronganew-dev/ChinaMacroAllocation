"""
基础算子抽象 — 量化算子的抽象基类约定

所有算子应继承 TimeSeriesOperator，统一接口：
- fit(series) → 内部状态
- transform(series) → 结果
- fit_transform(series) → 链式调用
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class TimeSeriesOperator(ABC):
    """时序算子抽象基类"""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._fitted = False

    @abstractmethod
    def fit(self, series: pd.Series) -> "TimeSeriesOperator":
        """学习序列所需的状态参数"""
        ...

    @abstractmethod
    def transform(self, series: pd.Series) -> Any:
        """对序列施加算子逻辑，返回计算结果"""
        ...

    def fit_transform(self, series: pd.Series) -> Any:
        """拟合并转换"""
        return self.fit(series).transform(series)

    def __repr__(self) -> str:
        return f"<{self.name}>"
