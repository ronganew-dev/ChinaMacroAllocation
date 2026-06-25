import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import models.utils

def run_step2_research():
    print("==========================================================")
    print("  启动: HP Filter 处理与宏观指标解释度研究 (Step 2)")
    print("==========================================================")
    
    from data_hub import DataHub
    data_hub = DataHub()
    universe = data_hub.get_universe()
    
    # 动态载入资产池
    base_assets = universe.base_risky_assets + universe.include_list
    rets_all = data_hub.get_returns(base_assets)
    risky_assets = universe.get_risky_assets(rets_all)
    
    # 1. 载入并处理宏观因子
    print("[*] 载入月频宏观数据并处理...")
    df_macro = data_hub.get_macro_data()
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    
    # 计算4个宏观因子序列
    raw_pmi = df_macro['PMI']
    raw_cpi = df_macro['CPI']
    raw_spread = df_macro['CN10Y'] - df_macro['US10Y']
    raw_yield = df_macro['CN10Y']
    
    raw_factors = {
        'PMI_score': raw_pmi,
        'CPI_score': raw_cpi,
        'SPREAD_score': raw_spread,
        'YIELD_score': raw_yield
    }
    
    factor_scores = {}
    twma_window = 6
    
    for name, raw_series in raw_factors.items():
        diffed = pd.Series(raw_series).diff(12).fillna(0.0)
        
        min_val = diffed.min()
        max_val = diffed.max()
        if max_val != min_val:
            scaled = (diffed - min_val) / (max_val - min_val)
        else:
            scaled = diffed * 0.0 + 0.5
            
        twma = scaled.rolling(window=twma_window, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        factor_scores[name] = twma
        
    df_factors = pd.DataFrame(factor_scores)
    df_factors['Date'] = df_macro['Date']
    df_factors['YM'] = df_factors['Date'].dt.to_period('M')
    
    # 2. 载入日频收益率并月度聚合
    df_ret = data_hub.get_returns(risky_assets)
    df_ret['Date'] = pd.to_datetime(df_ret['Date'])
    df_ret.set_index('Date', inplace=True)
    
    df_ret_m = df_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1).reset_index()
    df_ret_m['YM'] = df_ret_m['Date'].dt.to_period('M')
    
    # 3. 对齐合并
    df_merged = pd.merge(df_factors, df_ret_m, on='YM', suffixes=('_macro', '_ret'))
    df_merged.set_index('Date_macro', inplace=True)
    df_merged.index.name = 'Date'
    factor_cols = ['PMI_score', 'CPI_score', 'SPREAD_score', 'YIELD_score']
    
    # 计算每个资产的 WMA 收益率
    for asset in risky_assets:
        ret_series = df_merged[asset]
        wma_ret = ret_series.rolling(window=twma_window, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        df_merged[f'{asset}_WMA'] = wma_ret
        
    # 定义自变量特征 X (宏观因子当期值)
    X = df_merged[factor_cols]
    X_with_const = sm.add_constant(X)
    
    explanation_results = []
    
    # 针对两个模型 target 进行检验：
    # 1. Target = Next Month WMA Return (WMA模型)
    # 2. Target = Next Month Raw Return (Raw模型)
    for target_type in ['WMA', 'Raw']:
        print(f"\n--- [测试基于预测 {target_type} 收益率建立的多元回归模型] ---")
        
        for asset in risky_assets:
            # 建立下月收益率与宏观因子回归
            if target_type == 'WMA':
                y_target = df_merged[f'{asset}_WMA'].shift(-1)
            else:
                y_target = df_merged[asset].shift(-1)
                
            # 剔除由于 shift(-1) 产生的最后一个 NaN
            valid_mask = y_target.notna()
            X_valid = X_with_const[valid_mask]
            y_valid = y_target[valid_mask]
            
            # 全样本拟合
            model = sm.OLS(y_valid, X_valid).fit()
            
            # 使用估计的回归系数生成当期的宏观合成拟合指标 I_t
            # I_t = beta_0 + beta_1*F_1,t + beta_2*F_2,t + ...
            # 作用于整个数据集，不需要 drop 最后一期，确保实时生成最新宏观拟合指标
            composite_indicator = np.dot(X_with_const, model.params)
            df_merged[f'{asset}_I'] = composite_indicator
            
            # 对 composite_indicator 进行单向 HP filter 处理
            # lamb = 14400 是月频数据的经典平滑参数
            hp_trend = models.utils.one_sided_hp_filter(composite_indicator, lamb=14400)
            df_merged[f'{asset}_I_HP'] = hp_trend
            
            # 计算该 HP 过滤后的指标 I_HP_t 对：
            # 1) 当期风险资产收益率 R_A,t 的解释度
            # 2) 下一期风险资产收益率 R_A,t+1 的解释度
            
            # A. 解释当期收益率 R_A,t
            # 建立简单回归：R_A,t = alpha + gamma * I_HP_t
            y_curr = df_merged[asset]
            X_curr = sm.add_constant(df_merged[f'{asset}_I_HP'])
            res_curr = sm.OLS(y_curr, X_curr).fit()
            r2_curr = res_curr.rsquared
            corr_curr = np.corrcoef(y_curr, df_merged[f'{asset}_I_HP'])[0, 1]
            
            # B. 解释下一期收益率 R_A,t+1
            y_next = df_merged[asset].shift(-1)
            valid_mask_next = y_next.notna()
            y_next_v = y_next[valid_mask_next]
            X_next_v = sm.add_constant(df_merged[f'{asset}_I_HP'][valid_mask_next])
            res_next = sm.OLS(y_next_v, X_next_v).fit()
            r2_next = res_next.rsquared
            corr_next = np.corrcoef(y_next_v, df_merged[f'{asset}_I_HP'][valid_mask_next])[0, 1]
            
            explanation_results.append({
                'Target_Model': target_type,
                'Asset': asset,
                'Current_R2': r2_curr,
                'Current_Corr': corr_curr,
                'Next_R2': r2_next,
                'Next_Corr': corr_next
            })
            
    df_res = pd.DataFrame(explanation_results)
    
    # 输出汇总表
    print("\n\n" + "="*80)
    print("                     HP 滤波过滤后宏观拟合指标对收益率解释度对比表")
    print("="*80)
    for target_type in ['WMA', 'Raw']:
        print(f"\n[多元回归模型 Target: {target_type}]")
        sub_df = df_res[df_res['Target_Model'] == target_type].drop(columns=['Target_Model'])
        print(sub_df.to_string(index=False, formatters={
            'Current_R2': '{:.2%}'.format,
            'Current_Corr': '{:.4f}'.format,
            'Next_R2': '{:.2%}'.format,
            'Next_Corr': '{:.4f}'.format
        }))
    print("="*80)

if __name__ == '__main__':
    run_step2_research();
