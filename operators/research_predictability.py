import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import shutil

# Set matplotlib style for nice graphics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Heiti TC', 'SimHei']  # Support Chinese characters
plt.rcParams['axes.unicode_minus'] = False

def run_step1_research():
    print("==========================================================")
    print("  启动: 宏观因子对下月股票收益率的 Predictability 研究 (Step 1)")
    print("==========================================================")
    
    from data_hub import DataHub
    data_hub = DataHub()
    universe = data_hub.get_universe()
    
    # 动态载入资产池
    base_assets = universe.base_risky_assets + universe.include_list
    rets_all = data_hub.get_returns(base_assets)
    risky_assets = universe.get_risky_assets(rets_all)
    
    output_dir = os.path.join('data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 载入并处理宏观因子
    print("[*] 载入月频宏观数据...")
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
    
    # 进行 统一数理流水线处理 (YoY Diff -> MinMax to [0,1] -> 6M WMA)
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
    
    # 2. 载入并处理日频收益率
    print("[*] 载入日频资产收益率并进行月度聚合...")
    df_ret = data_hub.get_returns(risky_assets)
    df_ret['Date'] = pd.to_datetime(df_ret['Date'])
    df_ret.set_index('Date', inplace=True)
    
    # 转换为月度收益率 (1 + r).prod() - 1
    df_ret_m = df_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1).reset_index()
    df_ret_m['YM'] = df_ret_m['Date'].dt.to_period('M')
    
    # 3. 双表按 YM 对齐合并
    df_merged = pd.merge(df_factors, df_ret_m, on='YM', suffixes=('_macro', '_ret'))
    df_merged.set_index('Date_macro', inplace=True)
    df_merged.index.name = 'Date'
    factor_cols = ['PMI_score', 'CPI_score', 'SPREAD_score', 'YIELD_score']
    
    # 计算每个资产近 6 个月时间加权移动平均值 (WMA) 收益率
    for asset in risky_assets:
        ret_series = df_merged[asset]
        wma_ret = ret_series.rolling(window=twma_window, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        df_merged[f'{asset}_WMA'] = wma_ret
        
    # 对齐时间：将因子 X_t 与下个月的收益率 Y_{t+1} 进行对齐
    # Y_{t+1} 可以是 Raw Return 下月值，或者 WMA Return 下月值
    for asset in risky_assets:
        df_merged[f'{asset}_next_ret'] = df_merged[asset].shift(-1)
        df_merged[f'{asset}_next_WMA'] = df_merged[f'{asset}_WMA'].shift(-1)
        
    # 剔除最后一期 (由于 shift(-1) 导致其 Y 为 NaN)
    df_analysis = df_merged.dropna(subset=[f'{asset}_next_ret' for asset in risky_assets]).copy()
    
    print(f"[*] 对齐分析时间跨度: {df_analysis.index.min().strftime('%Y-%m-%d')} 至 {df_analysis.index.max().strftime('%Y-%m-%d')} (共 {len(df_analysis)} 个月)")
    
    # 4. 分别针对 6 个风险资产进行回归分析与 Out-of-Sample 检验
    oos_results = []
    
    for asset in risky_assets:
        print(f"\n==========================================================")
        print(f"  资产分析：【{asset}】")
        print(f"==========================================================")
        
        # -----------------
        # A. 全样本 OLS 显著性检验 (预测 WMA 收益率)
        # -----------------
        X = sm.add_constant(df_analysis[factor_cols])
        y_wma = df_analysis[f'{asset}_next_WMA']
        y_raw = df_analysis[f'{asset}_next_ret']
        
        model_wma = sm.OLS(y_wma, X).fit()
        model_raw = sm.OLS(y_raw, X).fit()
        
        print("\n[全样本 OLS 回归结果 - 预测下一月 WMA 收益率]")
        print(model_wma.summary().tables[1])
        print(f"R-squared: {model_wma.rsquared:.4f}, Adj. R-squared: {model_wma.rsquared_adj:.4f}, F-pvalue: {model_wma.f_pvalue:.4e}")
        
        print("\n[全样本 OLS 回归结果 - 预测下一月 Raw 收益率]")
        print(model_raw.summary().tables[1])
        print(f"R-squared: {model_raw.rsquared:.4f}, Adj. R-squared: {model_raw.rsquared_adj:.4f}, F-pvalue: {model_raw.f_pvalue:.4e}")
        
        # -----------------
        # B. Out-of-Sample (OOS) 滚动/递增预测检验
        # -----------------
        # 初始在样本窗口 (In-sample window): 36个月
        init_win = 36
        n_samples = len(df_analysis)
        
        oos_preds_wma = []
        oos_preds_raw = []
        actuals_wma = []
        actuals_raw = []
        hist_means_wma = []
        hist_means_raw = []
        
        for t in range(init_win, n_samples):
            # 训练集: 0 至 t-1
            train_X = sm.add_constant(df_analysis[factor_cols].iloc[:t])
            train_y_wma = df_analysis[f'{asset}_next_WMA'].iloc[:t]
            train_y_raw = df_analysis[f'{asset}_next_ret'].iloc[:t]
            
            # 拟合 OLS
            fit_wma = sm.OLS(train_y_wma, train_X).fit()
            fit_raw = sm.OLS(train_y_raw, train_X).fit()
            
            # 测试集第 t 期的预测特征
            test_x = np.array([1.0] + df_analysis[factor_cols].iloc[t].tolist())
            
            # OOS 预测
            pred_wma = np.dot(test_x, fit_wma.params)
            pred_raw = np.dot(test_x, fit_raw.params)
            
            oos_preds_wma.append(pred_wma)
            oos_preds_raw.append(pred_raw)
            
            actuals_wma.append(df_analysis[f'{asset}_next_WMA'].iloc[t])
            actuals_raw.append(df_analysis[f'{asset}_next_ret'].iloc[t])
            
            # 历史均值作为 Baseline 预测
            hist_means_wma.append(train_y_wma.mean())
            hist_means_raw.append(train_y_raw.mean())
            
        # 计算 OOS R-squared (Campbell-Thompson)
        # R2_OOS = 1 - sum((y - y_pred)^2) / sum((y - y_mean)^2)
        actuals_wma = np.array(actuals_wma)
        oos_preds_wma = np.array(oos_preds_wma)
        hist_means_wma = np.array(hist_means_wma)
        
        mse_pred_wma = np.mean((actuals_wma - oos_preds_wma) ** 2)
        mse_base_wma = np.mean((actuals_wma - hist_means_wma) ** 2)
        r2_oos_wma = 1 - (mse_pred_wma / mse_base_wma)
        corr_oos_wma = np.corrcoef(actuals_wma, oos_preds_wma)[0, 1]
        
        actuals_raw = np.array(actuals_raw)
        oos_preds_raw = np.array(oos_preds_raw)
        hist_means_raw = np.array(hist_means_raw)
        
        mse_pred_raw = np.mean((actuals_raw - oos_preds_raw) ** 2)
        mse_base_raw = np.mean((actuals_raw - hist_means_raw) ** 2)
        r2_oos_raw = 1 - (mse_pred_raw / mse_base_raw)
        corr_oos_raw = np.corrcoef(actuals_raw, oos_preds_raw)[0, 1]
        
        print(f"\n[样本外 Out-of-Sample (OOS) 预测效果评估]")
        print(f"  预测 WMA 收益率: OOS R2 = {r2_oos_wma:.2%}, OOS 预测与实际相关系数 = {corr_oos_wma:.4f}")
        print(f"  预测 Raw 收益率: OOS R2 = {r2_oos_raw:.2%}, OOS 预测与实际相关系数 = {corr_oos_raw:.4f}")
        
        oos_results.append({
            'Asset': asset,
            'FullSample_R2_WMA': model_wma.rsquared,
            'FullSample_R2_Raw': model_raw.rsquared,
            'OOS_R2_WMA': r2_oos_wma,
            'OOS_Corr_WMA': corr_oos_wma,
            'OOS_R2_Raw': r2_oos_raw,
            'OOS_Corr_Raw': corr_oos_raw
        })
        
        # -----------------
        # C. 绘制对齐时间了的宏观因子与 WMA 股票收益率的对比图
        # -----------------
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 绘制4个宏观因子 (左轴，已经在0-1区间)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['PMI (WMA)', 'CPI (WMA)', 'SPREAD (WMA)', 'YIELD (WMA)']
        for col, color, label in zip(factor_cols, colors, labels):
            ax1.plot(df_analysis.index, df_analysis[col], color=color, alpha=0.7, label=label, linewidth=1.5)
            
        ax1.set_ylabel('Processed Macro Factors (Scale: 0-1)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.tick_params(axis='y', labelsize=10)
        
        # 绘制 WMA 股票收益率 (右轴)
        ax2 = ax1.twinx()
        # 将 t+1 月 WMA 收益率对齐到 t 月 (代表 t 月因子预测 t+1 收益率的配对显示)
        ax2.plot(df_analysis.index, df_analysis[f'{asset}_next_WMA'], color='#9467bd', linewidth=2, linestyle='-', label=f'{asset} Next Month WMA Return')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_ylabel(f'{asset} Next Month WMA Return', fontsize=12, color='#9467bd')
        ax2.tick_params(axis='y', labelcolor='#9467bd', labelsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.1%}".format(x)))
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        
        plt.title(f'【{asset}】对齐的宏观因子 (6M WMA) 与 下月股票收益率 (6M WMA) 趋势图', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'step1_fit_{asset}.png')
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"    [成功] 已保存对齐趋势图至: {plot_path}")
        
    # 5. 输出汇总表
    print("\n\n" + "="*80)
    print("                      宏观因子回归与样本外(OOS)预测效果对比汇总表")
    print("="*80)
    df_summary = pd.DataFrame(oos_results)
    print(df_summary.to_string(index=False, formatters={
        'FullSample_R2_WMA': '{:.2%}'.format,
        'FullSample_R2_Raw': '{:.2%}'.format,
        'OOS_R2_WMA': '{:.2%}'.format,
        'OOS_Corr_WMA': '{:.4f}'.format,
        'OOS_R2_Raw': '{:.2%}'.format,
        'OOS_Corr_Raw': '{:.4f}'.format
    }))
    print("="*80)
    
    # 6. 将图片拷贝到 artifacts 目录，以便在 walkthrough 中展示
    art_dir = "/Users/charlottelu/.gemini/antigravity-ide/brain/6f5b1228-96b9-4558-803c-b603a03046ad"
    if os.path.exists(art_dir):
        print("\n[*] 拷贝生成的对比图至 artifacts 目录...")
        for asset in risky_assets:
            src = os.path.join(output_dir, f'step1_fit_{asset}.png')
            dst = os.path.join(art_dir, f'step1_fit_{asset}.png')
            shutil.copy(src, dst)
            print(f"    已拷贝 {asset} 对比图.")

if __name__ == '__main__':
    run_step1_research()
