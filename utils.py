import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook


def calc_ewma(data, decay_factor):
    """计算指数加权移动平均协方差矩阵"""
    n = data.shape[0]
    weights = np.array([(1 - decay_factor) * decay_factor ** i for i in range(n-1, -1, -1)])
    weights = weights / weights.sum()
    # 计算加权协方差矩阵
    mean = np.dot(weights, data)
    centered_data = data - mean
    weighted_centered_data = centered_data * np.sqrt(weights.reshape(-1, 1))
    cov_matrix = np.dot(weighted_centered_data.T, weighted_centered_data)
    return cov_matrix


def find_month_end(dates):
    """查找月末日期"""
    month_end = []
    for i in range(len(dates)):
        if i == 0:
            month_end.append(True)
        else:
            if dates[i].month != dates[i-1].month or dates[i].year != dates[i-1].year:
                month_end.append(True)
            else:
                month_end.append(False)
    return month_end


def calc_drawdowns(cumuValue, dates, isCompounding):
    """计算最大回撤"""
    nav = pd.Series(cumuValue, index=dates)
    if isCompounding:
        nav = nav / nav.iloc[0]
    max_nav = nav.expanding().max()
    if isCompounding:
        safe_max_nav = max_nav.replace(0, np.nan)
        drawdown = (nav / safe_max_nav - 1)
    else:
        # additive series (e.g. cumulative alpha) should use absolute drawdown
        drawdown = nav - max_nav
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan)
    max_drawdown = drawdown.min(skipna=True)
    peak_date = max_nav.idxmax()
    valid_drawdown = drawdown.dropna()
    trough_date = valid_drawdown.idxmin() if not valid_drawdown.empty else pd.NaT
    recovery_date = None
    
    # 处理 NaT 情况
    if pd.isna(trough_date):
        return {'peak': [0], 'peak_date': [dates[0]], 'trough_date': [dates[0]], 'recovery_date': [dates[0]]}
    
    try:
        for i in range(drawdown.index.get_loc(trough_date), len(drawdown)):
            if drawdown.iloc[i] >= 0:
                recovery_date = drawdown.index[i]
                break
    except (KeyError, ValueError):
        pass
    
    return {'peak': [max_drawdown], 'peak_date': [peak_date], 'trough_date': [trough_date], 'recovery_date': [recovery_date]}


def write_dataframes_to_excel(df_dict, file_path):
    """将多个 DataFrame 写入 Excel 文件"""
    if os.path.isfile(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
