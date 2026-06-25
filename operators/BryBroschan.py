import numpy as np
import pandas as pd

def run_realtime_macro_rotation(df, p_window=5, min_phase=5):
    """
    42个月基钦周期资产轮动系统 (开箱即用版)
    
    参数:
    ----------
    df : pd.DataFrame
        输入数据，索引为DatetimeIndex，必须包含 'sine_fitted' 列。
    p_window : int, 默认 5
        BB算法局部寻优窗口（前后看k个月）
    min_phase : int, 默认 5
        单边趋势最小持续月数
        
    返回:
    -------
    pd.DataFrame : 增加了斜率、加速度、象限(Regime)及资产配置权重的DataFrame
    """
    res_df = df.copy()
    
    # ==========================================
    # 步骤 1: 计算一阶导数(斜率)和二阶导数(加速度)
    # ==========================================
    # 用中央差分法平滑计算变化率
    res_df['slope'] = res_df['sine_fitted'].diff(1)
    res_df['acceleration'] = res_df['slope'].diff(1)
    
    # ==========================================
    # 步骤 2: 滚动模拟实时 BB 算法 (拒绝前瞻偏差)
    # ==========================================
    regimes = []
    
    for i in range(len(res_df)):
        # 严格隔离：只看当前及之前的数据
        current_history = res_df['sine_fitted'].iloc[:i+1]
        
        if len(current_history) < (2 * p_window + 1):
            regimes.append(np.nan) # 数据太短，无法初始化
            continue
            
        # 在历史数据中寻找局部的 Peaks 和 Troughs
        # 寻找极值点（必须在当前历史的内部，两端留出 window 距离）
        rolling_max = current_history.rolling(2 * p_window + 1, center=True).max()
        rolling_min = current_history.rolling(2 * p_window + 1, center=True).min()
        
        is_peak = (current_history == rolling_max)
        is_trough = (current_history == rolling_min)
        
        # 提取历史中已确认的所有拐点时间戳
        confirmed_peaks = current_history[is_peak].index
        confirmed_troughs = current_history[is_trough].index
        
        if len(confirmed_peaks) == 0 or len(confirmed_troughs) == 0:
            # 如果历史太短还没确认过完整的波峰波谷，退化为通过导数粗略判断
            last_slope = res_df['slope'].iloc[i]
            last_acc = res_df['acceleration'].iloc[i]
        else:
            # 找到离当前最近的一个确定的拐点
            last_peak_time = confirmed_peaks[-1]
            last_trough_time = confirmed_troughs[-1]
            
            # 判断当前处于上行大趋势还是下行大趋势
            # 即使 BB 算法有5个月滞后，我们也能知道当前在最新拐点的哪一侧
            if last_peak_time > last_trough_time:
                # 最新确认的是波峰 -> 处于下行大阶段 (BB_Regime = 0)
                # 结合当前实时加速度微调
                last_slope = res_df['slope'].iloc[i]
                last_acc = res_df['acceleration'].iloc[i]
            else:
                # 最新确认的是波谷 -> 处于上行大阶段 (BB_Regime = 1)
                last_slope = res_df['slope'].iloc[i]
                last_acc = res_df['acceleration'].iloc[i]

        # ==========================================
        # 步骤 3: 划分美林时钟四象限 (结合资产领先性避坑)
        # ==========================================
        # 象限 1: 复苏期 (斜率>0, 加速度>0) -> 经济见底回升，加速上行
        if last_slope >= 0 and last_acc >= 0:
            regime = "1_Recovery"
        # 象限 2: 过热期 (斜率>0, 加速度<0) -> 经济见顶前夕，增速放缓
        elif last_slope >= 0 and last_acc < 0:
            regime = "2_Overheating"
        # 象限 3: 滞胀期 (斜率<0, 加速度<0) -> 经济见顶回落，加速下行
        elif last_slope < 0 and last_acc < 0:
            regime = "3_Stagflation"
        # 象限 4: 衰退期 (斜率<0, 加速度>0) -> 经济筑底前夕，跌幅收窄
        else:
            regime = "4_Recession"
            
        regimes.append(regime)
        
    res_df['Regime'] = regimes

    # ==========================================
    # 步骤 4: 映射资产配置权重 (开箱即用策略)
    # ==========================================
    # 定义四个象限的标的资产权重 (考虑了权益资产前置领先性)
    weight_mapping = {
        "1_Recovery":    {"Stock": 0.60, "Commodity": 0.20, "Bond": 0.10, "Cash": 0.10}, # 超配股票
        "2_Overheating": {"Stock": 0.20, "Commodity": 0.60, "Bond": 0.00, "Cash": 0.20}, # 超配大宗商品
        "3_Stagflation": {"Stock": 0.10, "Commodity": 0.10, "Bond": 0.20, "Cash": 0.60}, # 超配现金防御
        "4_Recession":   {"Stock": 0.20, "Commodity": 0.00, "Bond": 0.70, "Cash": 0.10}  # 超配长端国债
    }
    
    # 初始化资产权重列
    for asset in ["Stock", "Commodity", "Bond", "Cash"]:
        res_df[f'w_{asset}'] = res_df['Regime'].map(lambda x: weight_mapping[x][asset] if pd.notna(x) else 0.25)
        
    return res_df


# ==========================================
# 3. 模拟测试运行 (如何调用)
# ==========================================
if __name__ == "__main__":
    # 创建一个模拟的 42 个月周期正弦波数据
    months = 120
    date_rng = pd.date_range(start='2016-01-31', periods=months, freq='M')
    
    # 42个月周期的标准正弦波，波幅带有一点向上的长期趋势(避免完全对称)
    t = np.arange(months)
    sine_wave = np.sin(2 * np.pi * t / 42) + 0.005 * t 
    
    # 组装输入 DataFrame
    input_df = pd.DataFrame(index=date_rng)
    input_df['sine_fitted'] = sine_wave
    
    # 运行轮动系统
    output_df = run_realtime_macro_rotation(input_df)
    
    # 查看最后10个月的配置切换结果
    print(output_df[['sine_fitted', 'Regime', 'w_Stock', 'w_Commodity', 'w_Bond']].tail(10))