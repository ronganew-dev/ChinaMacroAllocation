"""
Universe — 资产池管理器

从 config/asset_tickers.csv 读取资产列表，
支持动态筛选（排除列表 / 历史数据阈值）。
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_CSV = _PROJECT_ROOT / "config" / "asset_tickers.csv"


class Universe:
    """
    资产池管理器。

    Parameters
    ----------
    asset_csv : str or Path, optional
        资产配置 CSV 路径，默认 config/asset_tickers.csv
    risky_assets : list of str, optional
        风险资产名称列表，None 则从 CSV 读取所有非现金资产
    safe_asset : str, optional
        安全资产名称，默认 "中债新综合财富总值指数"
    exclude_list : list of str, optional
        需排除的资产
    min_history_ratio : float
        最低历史数据占比阈值
    """

    def __init__(
        self,
        asset_csv: Optional[str] = None,
        risky_assets: Optional[list] = None,
        safe_asset: str = "中债新综合财富总值指数",
        exclude_list: Optional[list] = None,
        min_history_ratio: float = 0.0,
    ):
        self.safe_asset = safe_asset
        self.exclude_list = exclude_list or []
        self.min_history_ratio = min_history_ratio

        csv_path = Path(asset_csv) if asset_csv else ASSET_CSV
        if csv_path.exists():
            cfg = pd.read_csv(csv_path)
            all_names = cfg["name"].tolist()
            cash_names = cfg[cfg["ticker"] == "M0220163"]["name"].tolist()
            self.base_risky_assets = risky_assets or [
                n for n in all_names
                if n != safe_asset and n not in cash_names
            ]
        else:
            self.base_risky_assets = risky_assets or []

    def get_risky_assets(self, returns_df: Optional[pd.DataFrame] = None) -> list:
        """
        获取当前风险资产池。

        Parameters
        ----------
        returns_df : pd.DataFrame, optional
            收益率 DataFrame，用于动态剔除历史数据不足的资产

        Returns
        -------
        list of str
            资产名称列表
        """
        assets = list(self.base_risky_assets)

        # 排除
        assets = [a for a in assets if a not in self.exclude_list]

        # 动态剔除历史数据不足的资产
        if returns_df is not None and self.min_history_ratio > 0.0:
            total = len(returns_df)
            filtered = []
            for a in assets:
                if a in returns_df.columns:
                    ratio = returns_df[a].notna().sum() / total
                    if ratio >= self.min_history_ratio:
                        filtered.append(a)
                    else:
                        logging.warning(
                            f"资产 【{a}】 被动态剔除: "
                            f"历史数据占比 {ratio:.2%}, 低于阈值 {self.min_history_ratio:.2%}"
                        )
                else:
                    logging.warning(f"资产 【{a}】 被动态剔除: 数据源中不存在")
            assets = filtered

        return assets

    def get_safe_asset(self) -> str:
        """获取避险底仓资产名称"""
        return self.safe_asset
