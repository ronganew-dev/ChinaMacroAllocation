# -*- coding: utf-8 -*-
"""
Module: macro_data_fetcher.py
Description: Wind EDB 宏观信用代理指标数据中台提取器 — Macro Data Fetcher
             独立运行的宏观信用数据下载工具。连接 Wind API，下载并清洗数据，
             并计算 "M1-PPI" 剪刀差，最后输出为 macro_credit_raw.csv。
             Edb指标代码由根目录下的 data_hub_config.json (中的 macro_credit_fetcher_mappings) 统一管理。
"""

import json
import os
import numpy as np
import pandas as pd
from WindPy import w

# ==========================================
# 1. 配置文件加载与常量初始化
# ==========================================
START_DATE = "2015-12-31"
OUTPUT_PATH = "macro_credit_raw.csv"

# 从项目根目录加载统一配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "data_hub_config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# 读取 EDB 代码映射关系
MACRO_TICKER_DICT = CONFIG["macro_credit_fetcher_mappings"]

def fetch_wind_macro_data(ticker_dict: dict, start_date: str) -> pd.DataFrame:
    """
    通过 Wind API 提取宏观时序并转换为标准 DataFrame
    """
    if not w.isconnected():
        w.start()
        
    tickers = list(ticker_dict.values())
    
    # 自动向前推算1年，用于计算同比变化率
    start_dt = pd.to_datetime(start_date)
    fetch_start_date = (start_dt - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # 调用 Wind EDB 宏观经济数据库接口
    error_code, wind_data = w.edb(tickers, fetch_start_date, end_date, "Fill=Previous", usedf=True)
    
    if error_code != 0:
        raise RuntimeError(f"Wind API 获取数据失败, 错误代码: {error_code}")
        
    # 重命名列名，实现代码对 Wind 代码的脱敏
    reverse_dict = {v: k for k, v in ticker_dict.items()}
    wind_data = wind_data.rename(columns=reverse_dict)
    
    # 强制将索引转换为标准 DateTimeIndex 并规范为月度频率 (Month End)
    wind_data.index = pd.to_datetime(wind_data.index)
    wind_data = wind_data.resample('ME').last()
    
    # 1. 计算 PPI 同比 (从当月环比累计重构，绕过权限限制)
    wind_data["PPI"] = ((1 + wind_data["PPI_MoM"] / 100).rolling(12).apply(np.prod, raw=True) - 1) * 100
    
    # 2. 计算企业存款余额同比
    wind_data["CORP_DEPOSIT"] = wind_data["CORP_DEPOSIT_RAW"].pct_change(12) * 100
    
    # 3. 计算社会融资规模存量同比
    wind_data["TSF"] = wind_data["TSF_RAW"].pct_change(12) * 100
    
    # 过滤出用户期望的起始日期之后的数据
    wind_data = wind_data[wind_data.index >= start_dt]
    
    return wind_data

def preprocess_and_save():
    """
    业务逻辑组合与复合指标算子化处理
    """
    print(">>> 正在连接 Wind API 并下载宏观信用数据...")
    df_raw = fetch_wind_macro_data(MACRO_TICKER_DICT, START_DATE)
    
    # 金融工程细节：计算复合剪刀差指标 "M1-PPI"
    df_raw["M1_PPI"] = df_raw["M1"] - df_raw["PPI"]
    
    # 筛选出合成信用因子最终所需的 5 个代理序列
    final_features = ["M1_PPI", "M2", "CORP_DEPOSIT", "TOTAL_LOAN", "TSF"]
    df_final = df_raw[final_features].copy()
    
    # 将百分比数值标准化（例如 Wind 返回的 11.5% 转换为 0.115）
    df_final = df_final / 100.0
    
    # 保存至本地数据中台统一接口
    df_final.to_csv(OUTPUT_PATH, encoding="utf-8")
    print(f">>> 数据下载及清洗完毕，已成功对齐并打印至: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_and_save()