import numpy as np

class Config:

    input_folder = './'
    output_folder = './'
    input_file = input_folder + 'allCycleInput.xlsx'
    output_file = output_folder + 'allCycleOutput.xlsx'
    return_sheet = 'return'
    sub_port_sheet = 'subPort'
    daa_prob_sheet = 'DAAProb'

    signal_sheet = 'signal'
    risky_asset_names = ['沪深300', '标普500','量化中性','黄金', '中证商品指数','国债30年', '信用债1-3年']
    safe_asset_names = ['信用债0-1年', '现金']
    asset_names = risky_asset_names + safe_asset_names
    regime_names = {'过热', '复苏', '滞胀', '衰退'}
    start_date = '2017/1/1'

    days_per_year = 365
    b_days_per_year = 252
    months_per_year = 12
    ewma_halflife_days = 63
    ewma_decay_factor = 0.5 ** (1 / ewma_halflife_days)
    ewma_rollingDays = 1260
    risk_scaler = 0.1
    target_risk = 0.02
    exAnte_risk_up_bound = 0.08

    monthEnd_rebalance = False
    dailyNoTcostRebalance = True
    tradeSize = 1.0
    threshold = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20, 0.20]  # threshhold weight difference to trigger rebalanced
    tradeDelay = 1
    transaction_cost = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001]  # one way
    
    # 经济状态相关配置
    regime_column = 'regime'  # 经济状态列名
    sub_port_target_risk = 0.02  # 子组合目标风险
    cash_weight = 0.05  # 现金权重

    adjusted_asset_name = '量化中性'
    adjusted_asset_mean = 0.055 # 量化中性资产的平均收益率
    vol_multiplier = 1.5
    performance_fee = 0.2
    return_adjustment = adjusted_asset_mean * (vol_multiplier - 1 + performance_fee) / b_days_per_year
    
    








