"""
基础工具 — Formulaic Operators 底层支持

不定义任何算子类。
Formulaic Operators 是纯函数式范式：
- 无类、无状态、无 fit/transform
- 每个函数 = 一个原子公式
- 输入 pandas Series/DataFrame → 输出同等类型

本模块仅提供：
- nanmask: 输入验证与自动过滤的装饰器
- 类型别名约定
"""

from functools import wraps
from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

# ── 类型别名 ──────────────────────────────────────────────
Series = pd.Series
Frame = pd.DataFrame

T = TypeVar("T")


def nanmask(
    min_periods: int = 1,
    drop_inf: bool = True,
) -> Callable:
    """
    装饰器：自动丢弃 NaN/Inf 后执行运算，
    并保持与输入对齐的索引。

    Parameters
    ----------
    min_periods : int
        要求最少有效观测数，不足则返回 NaN。
    drop_inf : bool
        是否在计算前剔除无限值。

    Examples
    --------
    >>> @nanmask(min_periods=20)
    ... def sharpe(s: pd.Series) -> float:
    ...     return s.mean() / s.std() * 252**0.5
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(s: pd.Series, *args, **kwargs) -> T:
            s = s.dropna()
            if drop_inf:
                s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < min_periods:
                return type(s)([np.nan], index=s.index[:1]) if isinstance(s, pd.Series) else np.nan
            return func(s, *args, **kwargs)
        return wrapper
    return decorator
