import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# 1. 构造模拟宏观经济数据（实际应用中替换为您的 DataFrame）
np.random.seed(42)
months = np.arange(1, 301)  # 300个月的数据（约25年）

# 模拟：趋势 + 100月周期 + 42月周期 + 随机噪声
trend = 0.05 * months  # 长期增长趋势
cycle_100 = 5 * np.sin(2 * np.pi * months / 100)
cycle_42 = 3 * np.sin(2 * np.pi * months / 42)
noise = np.random.normal(0, 1, len(months))
raw_data = trend + cycle_100 + cycle_42 + noise

df = pd.DataFrame({'Month': months, 'Value': raw_data})

# 2. 数据预处理：去趋势 (这里使用简单的线性拟合去趋势，实际可用HP滤波)
poly = np.polyfit(df['Month'], df['Value'], 1)
df['Trend'] = np.polyval(poly, df['Month'])
df['Detrended'] = df['Value'] - df['Trend']  # 去趋势后的残差（已自带中心化）

# 3. 傅里叶变换
N = len(df)
dt = 1  # 采样间隔为1个月
fft_values = fft(df['Detrended'].values)
frequencies = fftfreq(N, d=dt)

# 4. 频率过滤
# 计算目标周期对应的频率边界
# 短周期 (42个月左右，设范围为 35-50)
f_short_low = 1 / 50
f_short_high = 1 / 35

# 中周期 (100个月左右，设范围为 80-120)
f_medium_low = 1 / 120
f_medium_high = 1 / 80

# 创建频域掩码
fft_short = fft_values.copy()
fft_medium = fft_values.copy()

# 对正负频率同时进行过滤
fft_short[(abs(frequencies) < f_short_low) | (abs(frequencies) > f_short_high)] = 0
fft_medium[(abs(frequencies) < f_medium_low) | (abs(frequencies) > f_medium_high)] = 0

# 5. 逆傅里叶变换变回时域
df['Short_Cycle_42m'] = np.real(ifft(fft_short))
df['Medium_Cycle_100m'] = np.real(ifft(fft_medium))

# 6. 结果可视化
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df['Month'], df['Detrended'], label='Detrended Data', color='gray', alpha=0.6)
plt.title('Macroeconomic Data Analysis via Fourier Transform')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df['Month'], df['Medium_Cycle_100m'], label='Medium Cycle (~100 months)', color='blue')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df['Month'], df['Short_Cycle_42m'], label='Short Cycle (~42 months)', color='orange')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()