"""
宏观周期配置策略 — 工具函数（向后兼容桥接层）

所有核心实现已迁移至 operators/ 模块，此处仅做重导出。
"""

from operators.volatility import ewma_cov
from operators.calendar_utils import (
    find_month_end,
    write_dataframes_to_excel,
)

# 为了向后兼容
calc_ewma = ewma_cov
