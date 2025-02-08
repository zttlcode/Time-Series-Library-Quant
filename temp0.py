import pandas as pd


# 读取CSV文件
df = pd.read_csv('./data/backtest_bar_N_d.csv', parse_dates=['date'])  # 假设'时间'列包含日期时间，所以将其解析为datetime类型
# 确保索引不是时间列（如果已经是，可以跳过这一步）
df.set_index('date', inplace=True)

# 创建一个新的列OT，并初始化为NaN（对于除了最后一行以外的所有行）
df['OT'] = pd.NaT  # 使用NaT来表示时间类型的缺失值，或者你也可以使用float的NaN
# 对于除了最后一行以外的所有行，设置OT为下一行的收盘价
for i in range(len(df) - 1):
    df.iloc[i, df.columns.get_loc('OT')] = df.iloc[i + 1, df.columns.get_loc('close')]
# 最后一行的OT设置为NaN（或者其他默认值，如0）
df.iloc[-1, df.columns.get_loc('OT')] = df.iloc[-1, df.columns.get_loc('close')]  # 或者使用float的NaN，或者其他默认值

# 如果需要，将索引重置回默认的整数索引
df.reset_index(inplace=True)

# 如果需要，将结果保存回CSV文件
df.to_csv('./data/backtest_bar_N_d2.csv', index=False)  # 不保存索引
