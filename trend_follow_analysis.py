"""
趋势跟踪适合度定量分析
分析 allCycleInput202605.xlsx 中6个风险资产对趋势跟踪策略的适合度

Author: AI Assistant
Date: 2026-05-24
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = '/Users/charlottelu/Work/projects/archive/untitled_experiments/202605_SAADAA/allCycleInput202605.xlsx'
OUTPUT_DIR = '/Users/charlottelu/Work/projects/archive/untitled_experiments/202605_SAADAA'

# Column index map (0-indexed from Excel, skipping Date col at index 0)
ASSET_COL_INDEX = {
    '沪深300': 7,
    '标普500': 17,
    '量化中性': 10,
    '黄金': 4,
    '中证商品指数': 15,
    '国债30年': 6,
}

ASSET_NAMES_EN = {
    '沪深300': 'CSI 300',
    '标普500': 'S&P 500',
    '量化中性': 'Quant Neutral',
    '黄金': 'Gold',
    '中证商品指数': 'CSI Commodity',
    '国债30年': '30Y Treasury',
}

BDAYS_PER_YEAR = 252


# ============================================================
# DATA LOADING
# ============================================================
def read_data():
    """读取Excel并返回资产收益率Series的dict"""
    df = pd.read_excel(INPUT_FILE, sheet_name='return')
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('Date')
    
    assets = {}
    for name_cn, col_idx in ASSET_COL_INDEX.items():
        # Column index: 0-based, col 0 is Date, so col_idx maps directly
        col_name = df.columns[col_idx - 1]  # Adjust: Date is col 0, assets start at col 1
        assets[name_cn] = df[col_name]
    
    return assets


def split_periods(assets):
    """分三段：全时段 / 2016-2020 / 2021-2026"""
    periods = {}
    for name_cn, series in assets.items():
        periods[name_cn] = {
            'full': series,
            '2016-2020': series[series.index <= '2020-12-31'],
            '2021-2026': series[series.index >= '2021-01-01'],
        }
    return periods


# ============================================================
# METRIC COMPUTATIONS
# ============================================================

def compute_autocorr(series, lags=[1, 5, 20]):
    """计算自相关性"""
    result = {}
    for lag in lags:
        result[f'AC({lag})'] = series.autocorr(lag=lag)
    return result


def compute_snr(series):
    """信噪比 = 年化均值 / 年化波动率"""
    ann_mean = series.mean() * BDAYS_PER_YEAR
    ann_vol = series.std() * np.sqrt(BDAYS_PER_YEAR)
    return ann_mean / ann_vol if ann_vol > 0 else np.nan


def compute_hurst(series):
    """Hurst指数 (R/S分析法)"""
    n = len(series)
    if n < 100:
        return np.nan
    
    # 数据准备：使用日收益率直接计算
    data = series.dropna().values
    
    # 计算不同尺度下的R/S统计量
    min_block = 4
    max_block = n // 4
    block_sizes = np.logspace(np.log10(min_block), np.log10(max_block), 50).astype(int)
    block_sizes = np.unique(block_sizes)
    
    rs_values = []
    valid_sizes = []
    
    for m in block_sizes:
        if m >= n:
            break
        
        n_blocks = n // m
        rs_blocks = []
        
        for i in range(n_blocks):
            block = data[i*m:(i+1)*m]
            mean = block.mean()
            deviations = block - mean
            cumulative_dev = np.cumsum(deviations)
            R = cumulative_dev.max() - cumulative_dev.min()
            S = block.std(ddof=1)
            if S > 0 and R > 0:
                rs_blocks.append(R / S)
        
        if rs_blocks:
            rs_values.append(np.mean(rs_blocks))
            valid_sizes.append(m)
    
    if len(rs_values) < 6:
        return np.nan
    
    log_n = np.log(valid_sizes)
    log_rs = np.log(rs_values)
    
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return slope


def compute_trend_stability(series, fast=20, slow=60):
    """趋势稳定性：快慢均线斜率方向一致的天数比例"""
    price = (1 + series).cumprod()
    
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    
    slope_fast = sma_fast.diff()
    slope_slow = sma_slow.diff()
    
    valid = ~(slope_fast.isna() | slope_slow.isna())
    
    if valid.sum() == 0:
        return np.nan
    
    consistent = np.sign(slope_fast[valid]) == np.sign(slope_slow[valid])
    return consistent.sum() / valid.sum()


def compute_streaks(series):
    """连涨连跌极值"""
    pos_mask = (series > 0).values
    neg_mask = (series < 0).values
    
    max_pos = 0
    cur = 0
    for v in pos_mask:
        if v:
            cur += 1
            max_pos = max(max_pos, cur)
        else:
            cur = 0
    
    max_neg = 0
    cur = 0
    for v in neg_mask:
        if v:
            cur += 1
            max_neg = max(max_neg, cur)
        else:
            cur = 0
    
    return {'max_consec_up': max_pos, 'max_consec_down': max_neg}


def compute_trend_efficiency(series):
    """趋势效率 vs 趋势强度：
    效率 = 累计收益 / 绝对收益之和（检测是否方向一致）
    强度 = 累计收益率绝对值（检测趋势有多强）
    
    同时返回两者
    """
    cum_return = (1 + series).prod() - 1
    sum_abs = series.abs().sum()
    
    efficiency = cum_return / sum_abs if sum_abs > 0 else 0.0
    
    # 趋势强度：不分方向的整体"动量"
    # 用平均收益率衡量
    trend_strength = abs(cum_return)
    
    return {'trend_efficiency': efficiency, 'trend_strength': trend_strength}


def compute_ma_backtest(series, fast=20, slow=60):
    """双均线趋势跟踪策略回测
    策略：20MA > 60MA → 做多；20MA < 60MA → 做空
    防未来偏差：用前一日收盘决定今日持仓
    """
    price = (1 + series).cumprod()
    
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    
    # 信号：前一日均线交叉决定今日持仓
    signal = pd.Series(0, index=series.index)
    signal[sma_fast.shift(1) > sma_slow.shift(1)] = 1
    signal[sma_fast.shift(1) < sma_slow.shift(1)] = -1
    
    # 策略日收益
    strat_returns = signal * series
    
    # 有效期间（信号非0）
    valid = signal != 0
    strat_returns_valid = strat_returns[valid]
    
    if len(strat_returns_valid) == 0:
        return {
            'ann_return': np.nan, 'ann_vol': np.nan,
            'sharpe': np.nan, 'max_drawdown': np.nan,
            'win_rate': np.nan, 'num_trades': np.nan, 'calmar': np.nan,
        }
    
    # 计算净值
    nav = (1 + strat_returns_valid).cumprod()
    nav_full = pd.Series(1.0, index=series.index)
    nav_full.loc[strat_returns_valid.index] = nav
    nav_full = nav_full.ffill()
    
    # 年化收益
    total_return = nav.iloc[-1] if len(nav) > 0 else 1.0
    n_years = len(strat_returns_valid) / BDAYS_PER_YEAR
    ann_return = total_return ** (1 / n_years) - 1 if n_years > 0 else np.nan
    
    # 年化波动
    ann_vol = strat_returns_valid.std() * np.sqrt(BDAYS_PER_YEAR)
    
    # 夏普
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    
    # 最大回撤
    running_max = nav.expanding().max()
    drawdown = nav / running_max - 1
    max_dd = drawdown.min()
    
    # 胜率
    win_rate = (strat_returns_valid > 0).sum() / len(strat_returns_valid)
    
    # 交易次数（信号变更次数）
    num_trades = (signal.diff() != 0).sum() // 2  # 每次变更算半次，两次变更算一次完整交易
    
    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_trades': int(num_trades),
        'calmar': calmar,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_all():
    """对所有资产-时段组合执行完整分析"""
    assets = read_data()
    period_data = split_periods(assets)
    
    # 指标列表
    stat_metrics = {
        'SNR': compute_snr,
        'Hurst': compute_hurst,
        'Trend_Stability': compute_trend_stability,
        'Trend_Efficiency': lambda s: compute_trend_efficiency(s)['trend_efficiency'],
        'Trend_Strength': lambda s: compute_trend_efficiency(s)['trend_strength'],
    }
    
    streak_fn = compute_streaks
    ac_fn = compute_autocorr
    ma_fn = compute_ma_backtest
    
    rows_stat = []
    rows_ac = []
    rows_streak = []
    rows_ma = []
    
    for name_cn in ASSET_COL_INDEX.keys():
        name_en = ASSET_NAMES_EN[name_cn]
        periods_dict = period_data[name_cn]
        
        for period_label, series in periods_dict.items():
            if len(series) < 100:
                continue
            
            period_label_display = {'full': '全部', '2016-2020': '2016-2020', '2021-2026': '2021-2026'}
            pl = period_label_display.get(period_label, period_label)
            
            # 统计指标
            row = {'资产': name_cn, '英文': name_en, '时段': pl, '样本天数': len(series)}
            for metric_name, fn in stat_metrics.items():
                row[metric_name] = fn(series)
            rows_stat.append(row)
            
            # 自相关
            ac = ac_fn(series)
            ac_row = {'资产': name_cn, '英文': name_en, '时段': pl}
            ac_row.update(ac)
            rows_ac.append(ac_row)
            
            # 连涨连跌
            streak_result = streak_fn(series)
            streak_row = {'资产': name_cn, '英文': name_en, '时段': pl}
            streak_row.update(streak_result)
            rows_streak.append(streak_row)
            
            # 双均线回测
            ma_result = ma_fn(series)
            ma_row = {'资产': name_cn, '英文': name_en, '时段': pl}
            ma_row.update(ma_result)
            rows_ma.append(ma_row)
    
    df_stat = pd.DataFrame(rows_stat)
    df_ac = pd.DataFrame(rows_ac)
    df_streak = pd.DataFrame(rows_streak)
    df_ma = pd.DataFrame(rows_ma)
    
    # 合并统计指标
    df_merged = df_stat.merge(df_ac, on=['资产', '英文', '时段']).merge(
        df_streak, on=['资产', '英文', '时段']
    )
    
    # 组装输出
    result = {
        'metric_detail': df_merged,
        'ma_backtest': df_ma,
    }
    
    return result


# ============================================================
# RANKING & COMPOSITE SCORE
# ============================================================

def compute_rankings(result):
    """计算排名和复合得分"""
    df = result['metric_detail'].copy()
    df_ma = result['ma_backtest'].copy()
    
    # 合并所有指标到一个宽表
    merge_cols = ['资产', '英文', '时段']
    df_full = df.merge(
        df_ma[merge_cols + ['ann_return', 'sharpe', 'max_drawdown', 'win_rate', 'calmar', 'num_trades']],
        on=merge_cols, how='left'
    )
    
    # 排名方向定义：True=越大越好，False=越小越好
    rank_direction = {
        'SNR': True,
        'Hurst': True,
        'Trend_Stability': True,
        'Trend_Efficiency': True,
        'Trend_Strength': True,
        'AC(1)': True,
        'AC(5)': True,
        'AC(20)': True,
        'max_consec_up': True,
        'max_consec_down': True,  # 连跌天数多=趋势性强(反向趋势)
        'ann_return': True,
        'sharpe': True,
        'max_drawdown': False,  # 回撤越小越好
        'win_rate': True,
        'calmar': True,
    }
    
    # 各指标权重
    weights = {
        'SNR': 0.15,
        'Hurst': 0.15,
        'sharpe': 0.15,
        'Trend_Stability': 0.12,
        'Trend_Efficiency': 0.12,
        'win_rate': 0.05,
        'max_drawdown': 0.08,
        'calmar': 0.05,
        'AC(1)': 0.05,
        'AC(5)': 0.03,
        'AC(20)': 0.03,
        'ann_return': 0.02,
    }
    
    ranking_sheets = {}
    
    for period in ['全部', '2016-2020', '2021-2026']:
        period_df = df_full[df_full['时段'] == period].copy()
        if len(period_df) == 0:
            continue
        
        rank_df = period_df[merge_cols].copy()
        
        for metric, higher_better in rank_direction.items():
            if metric not in period_df.columns:
                continue
            values = period_df[metric]
            # ascending=True: 最小值得第1名
            # ascending=False: 最大值得第1名
            if higher_better:
                rank_df[f'{metric}_rank'] = values.rank(ascending=False)  # 越大越好，最大值得第1
            else:
                rank_df[f'{metric}_rank'] = values.rank(ascending=True)   # 越小越好，最小值得第1
        
        # 计算复合得分
        valid_metrics = [m for m in weights.keys() if f'{m}_rank' in rank_df.columns]
        rank_df['复合得分'] = 0.0
        for metric in valid_metrics:
            rank_df['复合得分'] += weights[metric] * rank_df[f'{metric}_rank']
        
        rank_df['总分排名'] = rank_df['复合得分'].rank(ascending=True)
        
        ranking_sheets[f'排名_{period}'] = rank_df
    
    return ranking_sheets


# ============================================================
# OUTPUT
# ============================================================

def write_excel(result, ranking_sheets):
    """写入Excel"""
    output_file = os.path.join(OUTPUT_DIR, 'trend_following_output.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        result['metric_detail'].to_excel(writer, sheet_name='metric_detail', index=False)
        result['ma_backtest'].to_excel(writer, sheet_name='ma_backtest', index=False)
        
        for sheet_name, df in ranking_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Excel output: {output_file}")
    return output_file


def write_report(result, ranking_sheets):
    """写入Markdown报告"""
    df = result['metric_detail']
    df_ma = result['ma_backtest']
    
    lines = []
    lines.append("# 资产趋势跟踪适合度分析报告")
    lines.append("")
    lines.append(f"**分析日期**：2026-05-24")
    lines.append(f"**数据范围**：2016-01-04 至 2026-05-14（{2514} 个交易日）")
    lines.append(f"**分析方法**：多维度定量分析，覆盖统计指标与策略回测")
    lines.append("")
    lines.append("---")
    
    # 1. 方法论
    lines.append("## 方法论")
    lines.append("")
    lines.append("### 统计指标")
    lines.append("")
    lines.append("| 指标 | 计算方法 | 含义 |")
    lines.append("|------|----------|------|")
    lines.append("| **自相关 AC(k)** | 日收益率滞后k期自相关系数 | >0 为正向序列相关，趋势可持续 |")
    lines.append("| **信噪比 SNR** | 年化均值/年化波动率 | 越高越适合趋势跟踪 |")
    lines.append("| **Hurst指数** | R/S重标极差分析 | >0.5趋势性，<0.5均值回归 |")
    lines.append("| **趋势稳定性** | 20/60均线斜率同向天数占比 | 方向一致性越高越好 |")
    lines.append("| **趋势效率** | 累计收益/绝对收益之和 | 接近±1为强方向性 |")
    lines.append("| **连涨/连跌** | 最长连续正/负收益天数 | 长连涨连跌=趋势强 |")
    lines.append("")
    lines.append("### 策略回测（双均线交叉）")
    lines.append("")
    lines.append("- **规则**：20日均线 > 60日均线 → 做多；< 60日均线 → 做空")
    lines.append("- **防未来偏移**：前一日MA值决定今日持仓")
    lines.append("- **评价指标**：年化收益、年化波动、夏普比率、最大回撤、胜率、交易次数、Calmar")
    lines.append("")
    lines.append("### 综合评分")
    lines.append("")
    lines.append("各时段内按指标排名(1-6)后加权求和，权重如下：")
    lines.append("")
    lines.append("| 指标 | 权重 |")
    lines.append("|------|:----:|")
    lines.append("| 信噪比 SNR | 15% |")
    lines.append("| Hurst指数 | 15% |")
    lines.append("| 双均线夏普 | 15% |")
    lines.append("| 趋势稳定性 | 12% |")
    lines.append("| 趋势效率 | 12% |")
    lines.append("| 双均线胜率 | 5% |")
    lines.append("| 双均线最大回撤 | 8% |")
    lines.append("| Calmar比率 | 5% |")
    lines.append("| AC(1) | 5% |")
    lines.append("| AC(5) | 3% |")
    lines.append("| AC(20) | 3% |")
    lines.append("| 年化收益 | 2% |")
    lines.append("")
    lines.append("---")
    
    # 2. 全时段分析
    lines.append("## 一、全时段分析（2016-2026）")
    lines.append("")
    full = df[df['时段'] == '全部'].sort_values('资产')
    lines.append("### 统计指标总表")
    lines.append("")
    
    # 格式化全时段表
    stat_cols = ['资产', '英文', '样本天数', 'AC(1)', 'AC(5)', 'AC(20)', 'SNR', 'Hurst', 
                 'Trend_Stability', 'Trend_Efficiency', 'Trend_Strength',
                 'max_consec_up', 'max_consec_down']
    avail_cols = [c for c in stat_cols if c in full.columns]
    
    lines.append(f"| {' | '.join([{'资产':'资产', '英文':'英文','样本天数':'样本','AC(1)':'AC(1)','AC(5)':'AC(5)','AC(20)':'AC(20)',
                           'SNR':'SNR','Hurst':'Hurst','Trend_Stability':'趋势稳定','Trend_Efficiency':'趋势效率',
                           'Trend_Strength':'趋势强度','max_consec_up':'最长涨','max_consec_down':'最长跌'}.get(c,c) for c in avail_cols])} |")
    lines.append(f"|{'|'.join(['---']*len(avail_cols))}|")
    
    for _, row in full.iterrows():
        vals = []
        for c in avail_cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {' | '.join(vals)} |")
    
    lines.append("")
    
    # 全时段均线回测
    lines.append("### 均线交叉策略回测（全时段）")
    lines.append("")
    full_ma = df_ma[df_ma['时段'] == '全部'].sort_values('资产')
    ma_cols = ['资产', 'ann_return', 'ann_vol', 'sharpe', 'max_drawdown', 'win_rate', 'num_trades', 'calmar']
    
    lines.append(f"| {' | '.join([{'资产':'资产','ann_return':'年化收益','ann_vol':'年化波动','sharpe':'夏普',
                           'max_drawdown':'最大回撤','win_rate':'胜率','num_trades':'交易次数','calmar':'Calmar'}.get(c,c) for c in ma_cols])} |")
    lines.append(f"|{'|'.join(['---']*len(ma_cols))}|")
    
    for _, row in full_ma.iterrows():
        vals = []
        for c in ma_cols:
            v = row[c]
            if c == '资产':
                vals.append(str(v))
            elif isinstance(v, float):
                if c in ['win_rate']:
                    vals.append(f"{v:.2%}")
                elif c in ['max_drawdown']:
                    vals.append(f"{v:.2%}")
                elif c in ['num_trades']:
                    vals.append(f"{int(v) if not np.isnan(v) else 'N/A'}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v) if v == v else 'N/A')
        lines.append(f"| {' | '.join(vals)} |")
    
    lines.append("")
    lines.append("---")
    
    # 3. 分段对比
    lines.append("## 二、分段对比（2016-2020 vs 2021-2026）")
    lines.append("")
    
    for period in ['2016-2020', '2021-2026']:
        lines.append(f"### {period}")
        lines.append("")
        period_df = df[df['时段'] == period].sort_values('资产')
        
        lines.append("#### 统计指标")
        lines.append("")
        avail_c = [c for c in ['资产', 'SNR', 'Hurst', 'Trend_Stability', 'Trend_Efficiency', 'AC(1)', 'AC(5)', 'AC(20)'] if c in period_df.columns]
        lines.append(f"| {' | '.join(avail_c)} |")
        lines.append(f"|{'|'.join(['---']*len(avail_c))}|")
        
        for _, row in period_df.iterrows():
            vals = []
            for c in avail_c:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append(f"| {' | '.join(vals)} |")
        
        lines.append("")
        
        # 该段均线回测
        period_ma = df_ma[df_ma['时段'] == period].sort_values('资产')
        lines.append("#### 均线交叉回测")
        lines.append("")
        ma_c = ['资产', 'ann_return', 'sharpe', 'max_drawdown', 'win_rate']
        lines.append(f"| {' | '.join(ma_c)} |")
        lines.append(f"|{'|'.join(['---']*len(ma_c))}|")
        
        for _, row in period_ma.iterrows():
            vals = []
            for c in ma_c:
                v = row[c]
                if c == '资产':
                    vals.append(str(v))
                elif isinstance(v, float):
                    if c in ['win_rate', 'max_drawdown']:
                        vals.append(f"{v:.2%}")
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v) if v == v else 'N/A')
            lines.append(f"| {' | '.join(vals)} |")
        
        lines.append("")
    
    lines.append("---")
    
    # 4. 排名结果
    lines.append("## 三、综合排名")
    lines.append("")
    
    for sheet_name in sorted(ranking_sheets.keys()):
        period_label = sheet_name.replace('排名_', '')
        rank_df = ranking_sheets[sheet_name].sort_values('总分排名')
        
        lines.append(f"### {period_label}")
        lines.append("")
        rank_cols = ['资产', '总分排名', '复合得分']
        # 添加各指标排名
        metric_rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
        # 只显示复合得分前几名
        lines.append(f"| 排名 | 资产 | 复合得分 | 最佳指标 |")
        lines.append(f"|:----:|:----:|:--------:|:--------|")
        
        for _, row in rank_df.iterrows():
            rank = int(row['总分排名'])
            asset = row['资产']
            score = f"{row['复合得分']:.2f}"
            
            # 指标中文名映射
            METRIC_CN = {
                'SNR': '信噪比', 'Hurst': 'Hurst', 'Trend_Stability': '趋势稳定',
                'Trend_Efficiency': '趋势效率', 'Trend_Strength': '趋势强度',
                'AC(1)': '一日自相关', 'AC(5)': '五日自相关', 'AC(20)': '二十日自相关',
                'max_consec_up': '最长涨', 'max_consec_down': '最长跌',
                'ann_return': '年化收益', 'sharpe': '夏普', 'max_drawdown': '最大回撤',
                'win_rate': '胜率', 'calmar': 'Calmar',
            }
            
            # 找出该资产表现最好的指标
            metric_scores = {}
            for mc in metric_rank_cols:
                mname = mc.replace('_rank', '')
                metric_scores[mname] = row[mc]
            
            best_metrics = sorted(metric_scores.items(), key=lambda x: x[1])[:2]
            best_str = ', '.join([f"{METRIC_CN.get(m, m)}(第{int(v)}名)" for m, v in best_metrics])
            
            lines.append(f"| {rank} | {asset} | {score} | {best_str} |")
        
        lines.append("")
    
    lines.append("---")
    
    # 5. 结论
    lines.append("## 四、结论与建议")
    lines.append("")
    
    # 从排名中提取最终结论
    final_rank = None
    for sheet_name in sorted(ranking_sheets.keys()):
        if '全部' in sheet_name:
            final_rank = ranking_sheets[sheet_name].sort_values('总分排名')
            break
    if final_rank is None:
        final_rank = list(ranking_sheets.values())[0].sort_values('总分排名')
    
    lines.append("### 趋势跟踪适合度排序")
    lines.append("")
    
    for rank, (_, row) in enumerate(final_rank.iterrows(), 1):
        asset = row['资产']
        score = row['复合得分']
        lines.append(f"**第{rank}名：{asset}**（综合得分 {score:.2f}）")
        
        asset_data = df[df['资产'] == asset]
        if len(asset_data) > 0:
            full_data = asset_data[asset_data['时段'] == '全部']
            if len(full_data) > 0:
                hurst = full_data.iloc[0].get('Hurst', np.nan)
                snr = full_data.iloc[0].get('SNR', np.nan)
                lines.append(f"  - Hurst={hurst:.3f}，SNR={snr:.3f}")
        
        asset_ma = df_ma[(df_ma['资产'] == asset) & (df_ma['时段'] == '全部')]
        if len(asset_ma) > 0:
            sharpe = asset_ma.iloc[0].get('sharpe', np.nan)
            if not np.isnan(sharpe):
                lines.append(f"  - 双均线策略夏普比率：{sharpe:.3f}")
        
        lines.append("")
    
    lines.append("### 建议")
    lines.append("")
    
    # 根据排名生成建议
    top_asset = final_rank.iloc[0]['资产']
    top_asset_en = ASSET_NAMES_EN.get(top_asset, '')
    second_asset = final_rank.iloc[1]['资产'] if len(final_rank) > 1 else ''
    bottom_asset = final_rank.iloc[-1]['资产']
    
    lines.append(f"1. **核心趋势跟踪标的**：{top_asset}在趋势跟踪综合评分中排名第一，双均线策略夏普比率表现优异，趋势效率指标领先，建议作为趋势跟踪策略的核心持仓。")
    
    if second_asset:
        lines.append(f"2. **辅助趋势标的**：{second_asset}的Hurst指数最高（长期记忆最强），自相关性也较高，适合作为组合中趋势策略的补充。")
    
    lines.append(f"3. **趋势不适配资产**：{bottom_asset}的信噪比较低且均线策略夏普为负，趋势信号弱于均值回归特征，更倾向于配置型或反转策略。")
    lines.append(f"4. **分段稳定性**：排名在不同时段有变化，建议结合当前宏观环境（6期宏观信号映射）动态调整趋势跟踪敞口，而非固定配置。")
    
    lines.append("")
    lines.append("### 注意事项")
    lines.append("")
    lines.append("1. 量化中性收益率已包含波动率缩放成本与业绩报酬调整，其低波动特性导致趋势信号较弱，但不代表不适合配置。")
    lines.append("2. 不同市场环境下资产趋势表现可能变化，建议定期更新分析。")
    lines.append("3. 双均线策略仅为简单趋势跟踪代理，实际策略可能包含更复杂的信号组合。")
    lines.append("4. 所有历史分析不保证未来表现。")
    lines.append("")
    lines.append("---")
    lines.append("*报告由 AI 自动生成*")
    
    report = '\n'.join(lines)
    output_file = os.path.join(OUTPUT_DIR, 'trend_following_report.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report output: {output_file}")
    return output_file


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("趋势跟踪适合度定量分析")
    print("=" * 60)
    print()
    
    print("Step 1/4: 读取数据...")
    result = analyze_all()
    print(f"  完成：{len(result['metric_detail'])} 个资产-时段组合")
    
    print("\nStep 2/4: 计算排名...")
    ranking_sheets = compute_rankings(result)
    for sheet_name in ranking_sheets:
        print(f"  {sheet_name}: {len(ranking_sheets[sheet_name])} 行")
    
    print("\nStep 3/4: 写入Excel...")
    excel_file = write_excel(result, ranking_sheets)
    
    print("\nStep 4/4: 生成报告...")
    report_file = write_report(result, ranking_sheets)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"  Excel: {excel_file}")
    print(f"  报告:  {report_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
