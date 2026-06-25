# -*- coding: utf-8 -*-
"""
Description: 原子化 OECD 宏观合成指数算子引擎
"""

import numpy as np
import pandas as pd

class OECDSynthesizerOperator:
    """
    OECD 复合因子合成算子 (Stateless & Vectorized)
    """
    @staticmethod
    def calculate_sd(df: pd.DataFrame) -> pd.Series:
        """
        第二步：计算各指标的标准化偏差 (平均绝对离差)
        SD_j = Mean(|C_j(t) - Mean(C_j)|)
        """
        demean = df - df.mean()
        sd = demean.abs().mean()
        return sd

    @staticmethod
    def calculate_sc(df: pd.DataFrame, sd: pd.Series) -> pd.DataFrame:
        """
        第三步：计算各指标的标准化序列 (自动实现高噪声指标权重的自适应扣减)
        SC_j(t) = (C_j(t) - Mean(C_j)) / SD_j
        """
        demean = df - df.mean()
        sc = demean / sd
        return sc

    @classmethod
    def synthesize_ci(cls, df_metrics: pd.DataFrame, benchmark_series: pd.Series) -> pd.DataFrame:
        """
        第四步与第五步：横截面融合与基准量级变换 (向基准指标 X 对齐振幅与均值)
        """
        # 1. 计算各指标的标准化离差并转化为无量纲得分
        sd = cls.calculate_sd(df_metrics)
        sc = cls.calculate_sc(df_metrics, sd)
        
        # 2. 第四步：横截面等权求和 (融合多维信用代理信息)
        S = sc.sum(axis=1)
        
        # 3. 第五步：参照基准指标 X 进行数量级调整 (计算系数 k 和平移量 d)
        mean_X = benchmark_series.mean()
        mean_S = S.mean()
        
        # 计算振幅调节系数 k (波动的绝对值均值对齐)
        k = (benchmark_series - mean_X).abs().mean() / (S - mean_S).abs().mean()
        
        # 计算趋势平移量 d
        d = mean_X - mean_S
        
        # 4. 线性变换生成最终合成信用因子指数 CI
        CI = k * S + d
        
        # 封装为标准买方输出格式
        result = pd.DataFrame({
            "Raw_Synthesis_S": S,
            "Credit_Factor_CI": CI,
            "Benchmark_X": benchmark_series
        }, index=df_metrics.index)
        
        return result

def run_pipeline():
    import os
    from pathlib import Path
    
    # 动态获取项目根目录与文件路径
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CONFIG_PATH = PROJECT_ROOT / "config" / "macro_tickers.csv"
    CACHE_DIR = PROJECT_ROOT / "data" / "cache"
    
    print(">>> 正在从本地 Parquet 缓存加载 credit 类型数据...")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")
        
    tickers_df = pd.read_csv(CONFIG_PATH)
    credit_tickers = tickers_df[tickers_df["group"] == "credit"]["name"].tolist()
    
    dfs = []
    for name in credit_tickers:
        p = CACHE_DIR / f"{name}.parquet"
        if not p.exists():
            print(f"  ⚠️ 找不到缓存文件: {p}，跳过该指标。")
            continue
        
        df_ind = pd.read_parquet(p)
        df_ind["Date"] = pd.to_datetime(df_ind["Date"])
        # 强制将索引转换为标准 DateTimeIndex 并规范为月度频率
        df_ind = df_ind.set_index("Date").resample("ME").last().ffill()
        
        # 判断是绝对值还是百分比同比值
        is_absolute = df_ind[name].max() > 100
        if is_absolute:
            yoy = df_ind[name].pct_change(12)
        else:
            yoy = df_ind[name] / 100.0
            
        dfs.append(yoy)
        
    if not dfs:
        raise FileNotFoundError(f"在 {CACHE_DIR} 目录下没有加载到任何有效的信用 (credit) 数据 parquet 文件。")
        
    # 合并所有信用代理指标，规范化索引
    df_credit = pd.concat(dfs, axis=1, sort=True).dropna(how="all").sort_index()
    # 填充缺失值 (前向填充与后向填充)
    df_credit = df_credit.ffill().bfill()
    df_credit = df_credit.dropna()
    
    # 指定核心因子对齐的基准锚 (社会融资规模存量同比: SF_ratio)
    if "SF_ratio" in df_credit.columns:
        benchmark = df_credit["SF_ratio"].copy()
    else:
        benchmark = df_credit.iloc[:, 0].copy()
        
    # 执行 OECD 算子合成
    print(">>> 正在调用无状态 OECD 算子进行信用因子合成...")
    synthesis_engine = OECDSynthesizerOperator()
    df_synthesis = synthesis_engine.synthesize_ci(df_credit, benchmark)
    
    # =========================================================================
    # 买方量化级细节：正弦周期拟合算子计算因子拟合指标
    # =========================================================================
    print(">>> 正在调用正弦周期拟合算子计算因子拟合指标...")
    from operators.sin_fit import fit_sine_wave_rolling
    
    # 对合成的信用因子指标进行滚动正弦拟合
    df_synthesis = fit_sine_wave_rolling(df_synthesis, target_col="Credit_Factor_CI", window_size=60, period_prior=42)
    
    # =========================================================================
    # 买方量化硬核细节：时滞防护过滤器 (Look-ahead Filter)
    # 本地回测主引擎在读取此因子做美林时钟信号判断时，必须向后平移一期，规避发布时滞作弊。
    # =========================================================================
    df_synthesis["Credit_Factor_Signal_Lagged"] = df_synthesis["Credit_Factor_CI"].shift(1)
    
    # 打印部分结果以供校验
    print("\n[因子合成与拟合历史截面预览 - 满足无状态向量化标准]")
    print(df_synthesis.tail(10))
    
    # 输出最终因子矩阵
    output_factor_path = PROJECT_ROOT / "china_credit_factor_index.csv"
    df_synthesis.to_csv(output_factor_path, encoding="utf-8")
    print(f"\n>>> 信用因子及拟合指标合成成功！最终资产配置信号矩阵已打印至: {output_factor_path}")


if __name__ == "__main__":
    run_pipeline()