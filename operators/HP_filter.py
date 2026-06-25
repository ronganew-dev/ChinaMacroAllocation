import pandas as pd
import numpy as np
from scipy.sparse import linalg, eye, spdiags

def price_hp_filter(series: pd.Series, lamb: float = 1.0) -> pd.Series:
    """
    专门针对价格类指标（如国债收益率、商品价格）的低参数 HP 滤波算子
    作为前置数据清洗算子（Pre-processor）
    在不牺牲时效性的前提下，过滤高频微观杂音
    不像处理宏观数据 要寻找长周期拐点（需使用高 λ）
    参数:
    - series: pd.Series, 原始价格或收益率时序 (不能有缺失值)
    - lamb: float, 惩罚系数。买方清洗日频价格杂音通常取 1.0 ~ 10.0
    """
    s = series.dropna()
    N = len(s)
    if N < 3:
        return series
    
    # 构建 HP 滤波的差分矩阵
    I = eye(N)
    offsets = [0, 1, 2]
    data = np.repeat([[1, -2, 1]], N, axis=0).T
    D = spdiags(data, offsets, N-2, N)
    
    # 矩阵求逆求解趋势项: g = (I + lamb * D^T * D)^(-1) * y
    A = I + lamb * (D.T @ D)
    g = linalg.spsolve(A, s.values)
    
    return pd.Series(g, index=s.index)