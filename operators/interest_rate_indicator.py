import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti Light', 'Songti SC', 'Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据源路径
DATA_PATH = '../data/input/宏观指标.xlsx'
SHEET_NAME = '利率'

# 输出路径
OUTPUT_DIR = '../data/output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 定义7种政策工具及其关键字
POLICY_TOOLS = {
    '逆回购': ['逆回购利率'],
    'SLF': ['SLF'],
    'MLF': ['MLF'],
    '准备金率': ['人民币存款准备金率'],
    '准备金利率': ['超额存款准备金率(超储率):金融机构', '（TODO）法定准备金'],
    '贷款利率': ['LPR'],
    '存款利率': ['存款利率']
}

# 市场利率关键字（按优先级排序）
MARKET_RATE_KEYWORDS = ['DR007', 'SHIBOR', '国债到期收益率']


def load_data():
    """加载数据"""
    if not os.path.exists(DATA_PATH):
        print(f"错误：数据源文件 {DATA_PATH} 不存在")
        return None
    
    try:
        print(f"开始加载数据：{DATA_PATH}，工作表：{SHEET_NAME}")
        
        # 先读取整个工作表，查看其结构
        full_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
        print(f"工作表总行数：{len(full_df)}")
        print(f"前5行数据：")
        print(full_df.head())
        
        # 从第2行开始读取列名，跳过前4行元数据，从第6行开始读取数据
        # header=1 表示使用第2行（索引为1）作为列名
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=1, skiprows=[2, 3, 4])
        print(f"读取数据后总行数：{len(df)}")
        print(f"读取数据后前5行：")
        print(df.head())
        
        # 处理日期列（第一列）
        date_col = df.columns[0]
        print(f"日期列原始名称：{date_col}")
        df.rename(columns={date_col: '日期'}, inplace=True)
        
        # 检查日期列数据
        print(f"日期列数据类型：{df['日期'].dtype}")
        print(f"日期列前10个值：")
        print(df['日期'].head(10))
        
        # 过滤掉空值行
        print(f"过滤前总行数：{len(df)}")
        df = df.dropna(subset=[df.columns[0]])
        print(f"过滤空值后总行数：{len(df)}")
        
        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        print(f"日期转换后的数据类型：{df['日期'].dtype}")
        print(f"日期转换后前10个值：")
        print(df['日期'].head(10))
        
        # 过滤掉日期转换失败的行
        print(f"日期转换前总行数：{len(df)}")
        df = df.dropna(subset=['日期'])
        print(f"过滤日期转换失败后总行数：{len(df)}")
        
        # 检查日期范围
        if not df.empty:
            print(f"日期范围：{df['日期'].min()} 到 {df['日期'].max()}")
        
        # 只处理2005年之后到2025年12月的数据
        print(f"过滤前总行数：{len(df)}")
        df = df[(df['日期'] >= '2005-01-01') & (df['日期'] <= '2025-12-31')]
        print(f"2005-2025年总行数：{len(df)}")
        
        if not df.empty:
            df = df.sort_values('日期').reset_index(drop=True)
            print(f"排序后前5行日期：")
            print(df['日期'].head())
        
        # 清理列名，去除首尾空格
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # 打印实际列名，方便调试
        print("实际列名:")
        print(df.columns.tolist())
        
        # 检查每列的数据
        if not df.empty:
            print("各列数据统计：")
            for col in df.columns:
                if col != '日期':
                    non_null_count = df[col].count()
                    print(f"{col}: 非空值数量 = {non_null_count}")
        
        return df
    except Exception as e:
        print(f"加载数据时出错：{e}")
        import traceback
        traceback.print_exc()
        return None


def get_policy_tool_columns(df, tool_name, keywords):
    """获取政策工具对应的列"""
    columns = []
    for col in df.columns:
        if col == '日期':
            continue
        for keyword in keywords:
            if keyword in col:
                columns.append(col)
                break
    return columns


def calculate_policy_score(data, columns, window=3):
    """计算政策工具的打分"""
    scores = []
    
    # 检查是否还未出现第一个非零值
    def has_non_zero_value(data, columns, current_index):
        """检查当前索引及之前是否有非零值"""
        for i in range(current_index + 1):
            for col in columns:
                value = data.iloc[i][col]
                if pd.notna(value) and value != 0:
                    return True
        return False
    
    for i in range(len(data)):
        # 检查是否还未出现第一个非零值
        if not has_non_zero_value(data, columns, i):
            scores.append(0)
            continue
        
        if i < window - 1:
            scores.append(0)
            continue
        
        # 获取过去3个月的数据
        window_data = data.iloc[i-window+1:i+1]
        
        # 检查是否有下调或上调
        has_decrease = False
        has_increase = False
        
        for col in columns:
            col_data = window_data[col].dropna()
            if len(col_data) < 2:
                continue
            
            # 检查是否有下调
            if any(col_data.iloc[j] < col_data.iloc[j-1] for j in range(1, len(col_data))):
                has_decrease = True
            # 检查是否有上调
            if any(col_data.iloc[j] > col_data.iloc[j-1] for j in range(1, len(col_data))):
                has_increase = True
        
        # 根据规则打分
        if has_decrease and not has_increase:
            scores.append(1)
        elif has_increase and not has_decrease:
            scores.append(-1)
        else:
            scores.append(0)
    
    return scores


def calculate_market_rate_score(df, window=3):
    """计算市场利率的打分"""
    # 按照优先级提取市场利率列
    # 1. DR007
    # 2. SHIBOR
    # 3. 国债到期收益率
    
    # 提取各类型的列
    dr007_cols = [col for col in df.columns if 'DR007' in col]
    shibor_cols = [col for col in df.columns if 'SHIBOR' in col]
    bond_cols = [col for col in df.columns if '国债到期收益率' in col]
    
    print(f"找到DR007相关列：{dr007_cols}")
    print(f"找到SHIBOR相关列：{shibor_cols}")
    print(f"找到国债到期收益率相关列：{bond_cols}")
    
    # 创建填补后的市场利率数据
    market_data = pd.Series(index=df.index)
    
    # 优先使用DR007
    if dr007_cols:
        market_data = df[dr007_cols[0]].copy()
        print(f"使用DR007列作为基础：{dr007_cols[0]}")
    
    # 若有缺失值，使用SHIBOR填补
    if shibor_cols and market_data.isna().any():
        shibor_data = df[shibor_cols[0]]
        market_data = market_data.fillna(shibor_data)
        print(f"使用SHIBOR列填补缺失值：{shibor_cols[0]}")
    
    # 若还有缺失值，使用国债到期收益率填补
    if bond_cols and market_data.isna().any():
        bond_data = df[bond_cols[0]]
        market_data = market_data.fillna(bond_data)
        print(f"使用国债到期收益率列填补缺失值：{bond_cols[0]}")
    
    # 检查是否还有缺失值
    if market_data.isna().any():
        print("警告：市场利率数据仍有缺失值，将填充为0")
        market_data = market_data.fillna(0)
    
    scores = []
    
    for i in range(len(df)):
        if i < window - 1:
            scores.append(0)
            continue
        
        # 与3个月前的平滑值比较
        if i < 2 * window - 1:
            scores.append(0)
            continue
        
        # 3个月平滑处理（简单平均）
        current_avg = market_data.iloc[i-window+1:i+1].mean()
        previous_avg = market_data.iloc[i-2*window+1:i-window+1].mean()
        
        if pd.isna(current_avg) or pd.isna(previous_avg):
            scores.append(0)
            continue
        
        # 根据规则打分
        if current_avg < previous_avg:
            scores.append(1)
        elif current_avg > previous_avg:
            scores.append(-1)
        else:
            scores.append(0)
    
    return scores


def calculate_average_score(scores_dict):
    """计算平均打分"""
    # 将所有打分转换为DataFrame
    scores_df = pd.DataFrame(scores_dict)
    # 计算每行的平均值
    average_scores = scores_df.mean(axis=1).tolist()
    return average_scores


def plot_results(df, scores_dict, average_scores, money_factor):
    """绘制折线图"""
    plt.figure(figsize=(15, 8))
    
    # 绘制各政策工具打分
    for tool, scores in scores_dict.items():
        plt.plot(df['日期'], scores, label=tool, alpha=0.6)
    
    # 绘制平均打分
    plt.plot(df['日期'], average_scores, label='平均打分', linewidth=2, color='black')
    
    # 绘制货币因子
    plt.plot(df['日期'], money_factor, label='货币因子', linewidth=2, color='red', linestyle='--')
    
    plt.title('利率指标打分趋势', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('打分', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(OUTPUT_DIR, 'interest_rate_scores.png')
    plt.savefig(output_path)
    print(f"图表已保存到：{output_path}")
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    print("开始处理利率指标...")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    print(f"成功加载数据，数据量：{len(df)} 条")
    
    # 计算各政策工具的打分
    scores_dict = {}
    
    for tool_name, keywords in POLICY_TOOLS.items():
        columns = get_policy_tool_columns(df, tool_name, keywords)
        if not columns:
            print(f"警告：政策工具 '{tool_name}' 未找到对应列，将填充0分")
            scores_dict[tool_name] = [0] * len(df)
        else:
            print(f"政策工具 '{tool_name}' 找到 {len(columns)} 列：{columns}")
            scores_dict[tool_name] = calculate_policy_score(df, columns)
    
    # 计算市场利率的打分
    scores_dict['市场利率'] = calculate_market_rate_score(df)
    
    # 计算平均打分
    average_scores = calculate_average_score(scores_dict)
    
    # 生成货币因子信号
    money_factor = []
    for score in average_scores:
        if score > 0:
            money_factor.append(1)  # 货币宽松环境
        elif score < 0:
            money_factor.append(-1)  # 货币紧缩环境
        else:
            money_factor.append(0)  # 中性
    
    # 生成结果DataFrame
    result_df = df[['日期']].copy()
    for tool, scores in scores_dict.items():
        result_df[tool] = scores
    result_df['平均打分'] = average_scores
    result_df['货币因子'] = money_factor
    
    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, 'interest_rate_scores.xlsx')
    result_df.to_excel(output_file, index=False)
    print(f"结果已保存到：{output_file}")
    
    # 绘制图表
    plot_results(df, scores_dict, average_scores, money_factor)
    
    print("处理完成！")


if __name__ == "__main__":
    main()
