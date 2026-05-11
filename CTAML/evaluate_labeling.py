import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def evaluate_labeling(ohlcv_path="ohlcv.csv",
                      signals_labeled_path="signals_labeled.csv",
                      future_T=60):
    """
    评估标注结果：检查 D_label 与实际未来收益的关系

    参数:
        ohlcv_path: OHLCV 数据文件路径，必须包含 time 和 close 列
        signals_labeled_path: 标注后的信号文件，包含 time, price, signal, D_label, xi_label
        future_T: 未来窗口长度（与标注时使用的 future_T 一致，默认 60）
    """
    # 1. 读取数据
    ohlcv = pd.read_csv(ohlcv_path, parse_dates=['time'])
    ohlcv = ohlcv.set_index('time').sort_index()

    signals = pd.read_csv(signals_labeled_path, parse_dates=['time'])
    signals = signals.sort_values('time')

    # 只保留买入信号（因为 D 的定义针对买入，卖出的逻辑不同）
    buy_signals = signals[signals['signal'] == 'buy'].copy()
    print(f"总买入信号数: {len(buy_signals)}")
    print(f"其中 D=1.0 的数量: {(buy_signals['D_label'] == 1.0).sum()}")

    # 2. 对齐时间索引，确保 ohlcv 的时间范围覆盖信号时间
    # 确保 ohlcv 索引是 datetime 类型
    ohlcv.index = pd.to_datetime(ohlcv.index)
    buy_signals['time'] = pd.to_datetime(buy_signals['time'])

    # 3. 为每个信号计算未来 future_T 天内的最高价和最大收益率
    results = []
    for idx, row in buy_signals.iterrows():
        sig_time = row['time']
        sig_price = row['price']
        d_label = row['D_label']
        xi_label = row['xi_label']

        # 找到信号时间在 ohlcv 中的位置
        if sig_time not in ohlcv.index:
            # 如果信号时间恰好不在 OHLCV 索引中（比如周末），找最近的下一个交易日
            future_idx = ohlcv.index[ohlcv.index >= sig_time]
            if len(future_idx) == 0:
                continue
            sig_time = future_idx[0]

        # 提取信号之后 future_T 个交易日的数据（注意：未来 bar 数，不是自然日）
        loc = ohlcv.index.get_loc(sig_time)
        end_loc = min(loc + future_T, len(ohlcv) - 1)
        if loc >= end_loc:
            # 窗口不足，跳过
            continue

        future_prices = ohlcv['close'].iloc[loc + 1:end_loc + 1]
        if len(future_prices) == 0:
            max_return = np.nan
        else:
            max_price = future_prices.max()
            max_return = (max_price - sig_price) / sig_price

        results.append({
            'time': sig_time,
            'D_label': d_label,
            'xi_label': xi_label,
            'max_return': max_return
        })

    df_results = pd.DataFrame(results).dropna(subset=['max_return'])
    print(f"有效信号数（有足够未来数据）: {len(df_results)}")

    # 4. 按 D 分桶统计
    # 定义桶边界：0, 0.2, 0.4, 0.6, 0.8, 1.0（1.0 单独处理，因为 D=1.0 是特殊值）
    def bucket_D(d):
        if d == 1.0:
            return "1.0"
        elif d < 0.2:
            return "[0,0.2)"
        elif d < 0.4:
            return "[0.2,0.4)"
        elif d < 0.6:
            return "[0.4,0.6)"
        elif d < 0.8:
            return "[0.6,0.8)"
        else:
            return "[0.8,1.0)"

    df_results['D_bucket'] = df_results['D_label'].apply(bucket_D)

    # 按桶汇总
    stats = df_results.groupby('D_bucket').agg(
        count=('max_return', 'size'),
        mean_return=('max_return', 'mean'),
        median_return=('max_return', 'median'),
        win_rate=('max_return', lambda x: (x > 0).mean()),
        mean_xi=('xi_label', 'mean')
    ).reset_index()

    # 按照桶的顺序排序（小 D 在前）
    bucket_order = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)", "1.0"]
    stats['D_bucket'] = pd.Categorical(stats['D_bucket'], categories=bucket_order, ordered=True)
    stats = stats.sort_values('D_bucket')

    print("\n===== 按 D_label 分桶的统计结果 =====")
    print(stats.to_string(index=False))

    # 5. 计算 Spearman 相关系数（D 与 max_return）
    # 注意：D 越小应收益越高，所以期望负相关
    corr, p_value = spearmanr(df_results['D_label'], df_results['max_return'])
    print(f"\nSpearman 相关系数 (D_label vs 未来最大收益率): {corr:.4f}, p-value: {p_value:.4f}")
    if corr < 0 and p_value < 0.05:
        print("✅ 显著负相关，符合预期：D 越小，未来收益越高")
    elif corr > 0:
        print("⚠️ 正相关，说明标注与未来表现相反：D 越大的信号反而收益越高")
    else:
        print("⚠️ 无明显相关性，标注可能不够准确")

    # 6. 可选：画出箱线图 / 散点图
    plt.figure(figsize=(10, 6))
    # 箱线图：不同 D 桶的 max_return 分布
    sns.boxplot(data=df_results, x='D_bucket', y='max_return', order=bucket_order)
    plt.title('Future Max Return by D_bucket')
    plt.xticks(rotation=45)
    plt.ylabel('Max Return (future 60 days)')
    plt.xlabel('D_label bucket')
    plt.tight_layout()
    plt.savefig('labeling_eval_boxplot.png', dpi=150)
    print("箱线图已保存为 labeling_eval_boxplot.png")

    # 散点图：D vs max_return
    plt.figure(figsize=(8, 6))
    plt.scatter(df_results['D_label'], df_results['max_return'], alpha=0.3)
    plt.xlabel('D_label')
    plt.ylabel('Future Max Return')
    plt.title('Scatter plot: D_label vs Actual Return')
    # 添加趋势线
    z = np.polyfit(df_results['D_label'], df_results['max_return'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(df_results['D_label']), p(sorted(df_results['D_label'])), "r--")
    plt.savefig('labeling_eval_scatter.png', dpi=150)
    print("散点图已保存为 labeling_eval_scatter.png")

    return df_results, stats


if __name__ == "__main__":
    df_res, stats = evaluate_labeling()