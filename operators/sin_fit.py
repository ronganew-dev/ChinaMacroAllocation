import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def sine_model(t, A, omega, phi, C):
    """标准弦波数学模型"""
    return A * np.sin(omega * t + phi) + C

def fit_sine_wave_rolling(df, target_col, window_size=60, period_prior=42):
    """
    对宏观因子进行滚动窗口弦波拟合与相位提取
    
    参数:
    df: pd.DataFrame, 必须包含日期索引和因子列
    target_col: str, 拟合的目标列名 (如 'Credit_Factor_CI')
    window_size: int, 滚动窗口大小 (月频，默认60个月，覆盖至少一个完整基钦周期)
    period_prior: int, 周期的先验猜想值 (基钦周期默认42个月)
    """
    df = df.copy().sort_index()
    # 构建连续的时间自变量 t
    df['t_idx'] = np.arange(len(df))
    
    # 初始化存储结果的列
    df['Fitted_Value'] = np.nan
    df['Phase_Angle'] = np.nan  # 当前复合相位 (omega * t + phi) % (2*pi)
    df['Period_Months'] = np.nan # 拟合出的实际周期长度
    df['R_Squared'] = np.nan     # 拟合优度
    
    # 转换为标准的正弦初值猜想
    omega_guess = 2 * np.pi / period_prior
    
    # 滚动窗口计算 (从 window_size 开始避免样本不足)
    for i in range(window_size, len(df) + 1):
        window_data = df.iloc[i - window_size: i]
        current_idx = i - 1 # 当前月末对应的时点
        
        y_train = window_data[target_col].values
        t_train = window_data['t_idx'].values
        
        # 启发式初值设定，提高非线性最小二乘法的收敛稳定性
        A_guess = (np.max(y_train) - np.min(y_train)) / 2
        C_guess = np.mean(y_train)
        phi_guess = 0.0
        p0 = [A_guess, omega_guess, phi_guess, C_guess]
        
        # 参数边界约束：限制周期长度在 24 个月到 60 个月之间，防止拟合出极端无意义的波形
        lower_bounds = [0, 2 * np.pi / 60, -np.pi, -np.inf]
        upper_bounds = [np.inf, 2 * np.pi / 24, np.pi, np.inf]
        
        try:
            # 使用 Levenberg-Marquardt 或 Trust Region Reflective 算法拟合
            popt, _ = curve_fit(sine_model, t_train, y_train, p0=p0, 
                                bounds=(lower_bounds, upper_bounds), maxfev=5000)
            A_fit, omega_fit, phi_fit, C_fit = popt
            
            # 计算当前最新时点 t 对应的拟合值和相位
            t_curr = df.iloc[current_idx]['t_idx']
            fitted_val = sine_model(t_curr, A_fit, omega_fit, phi_fit, C_fit)
            
            # 提取复合相位，并将弧度标准化映射至 [0, 2*pi) 区间
            raw_phase = (omega_fit * t_curr + phi_fit) % (2 * np.pi)
            
            # 计算拟合优度 R^2
            y_pred = sine_model(t_train, A_fit, omega_fit, phi_fit, C_fit)
            ss_res = np.sum((y_train - y_pred) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 将当前时点的计算结果回填
            date_anchor = df.index[current_idx]
            df.loc[date_anchor, 'Fitted_Value'] = fitted_val
            df.loc[date_anchor, 'Phase_Angle'] = raw_phase
            df.loc[date_anchor, 'Period_Months'] = 2 * np.pi / omega_fit
            df.loc[date_anchor, 'R_Squared'] = r2
            
        except Exception as e:
            # 拟合失败时保持 NaN，便于后续清洗
            continue
            
    return df

# 使用示例 (假设你已载入 csv 并将日期设为了索引)
# data = pd.read_csv('china_credit_factor_index.csv', index_col=0, parse_dates=True)
# fitted_res = fit_sine_wave_rolling(data, target_col='Credit_Factor_CI')