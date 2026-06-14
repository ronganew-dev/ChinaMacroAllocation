import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Universe:
    def __init__(self, universe_config):
        self.config = universe_config
        self.base_risky_assets = self.config.get("risky_assets", [])
        self.safe_asset = self.config.get("safe_asset", "中债新综合财富总值指数")
        self.exclude_list = self.config.get("exclude_list", [])
        self.include_list = self.config.get("include_list", [])
        self.min_history_ratio = self.config.get("min_history_ratio", 0.0)

    def get_risky_assets(self, returns_df=None) -> list:
        """
        根据基础配置、包含列表、排除列表以及可选的动态历史数据量阈值，获取当前的风险资产池。
        """
        # 1. 基础资产池 + 额外包含资产
        assets = list(self.base_risky_assets)
        for asset in self.include_list:
            if asset not in assets:
                assets.append(asset)
                
        # 2. 动态剔除排除列表中的资产
        assets = [a for a in assets if a not in self.exclude_list]
        
        # 3. 动态剔除历史数据不足的资产
        if returns_df is not None and self.min_history_ratio > 0.0:
            filtered_assets = []
            for asset in assets:
                if asset in returns_df.columns:
                    # 计算非空、非零数据的占比
                    non_null_count = returns_df[asset].notna().sum()
                    total_count = len(returns_df)
                    ratio = non_null_count / total_count if total_count > 0 else 0.0
                    
                    if ratio >= self.min_history_ratio:
                        filtered_assets.append(asset)
                    else:
                        logging.warning(
                            f"资产 【{asset}】 被动态剔除: 历史数据占比为 {ratio:.2%}, 低于配置阈值 {self.min_history_ratio:.2%}"
                        )
                else:
                    logging.warning(f"资产 【{asset}】 被动态剔除: 数据源中不存在该资产列")
            assets = filtered_assets
            
        return assets

    def get_safe_asset(self) -> str:
        """获取避险底仓资产名称"""
        return self.safe_asset
