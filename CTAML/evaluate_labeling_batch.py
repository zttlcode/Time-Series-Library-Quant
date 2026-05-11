import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os


def evaluate_labeling(ohlcv_path, signals_labeled_path, future_T=60):
    """
    评估单只股票的标注结果
    返回: (df_results, stats, corr, p_value)
        df_results: 每个信号的 D_label 和实际最大收益
        stats: 按 D_bucket 分组的统计数据
        corr: Spearman 相关系数
        p_value: 显著性
    """
    if not os.path.exists(ohlcv_path):
        print(f"OHLCV 文件不存在: {ohlcv_path}")
        return None, None, None, None
    if not os.path.exists(signals_labeled_path):
        print(f"标注信号文件不存在: {signals_labeled_path}")
        return None, None, None, None

    ohlcv = pd.read_csv(ohlcv_path, parse_dates=['time'])
    ohlcv = ohlcv.set_index('time').sort_index()

    signals = pd.read_csv(signals_labeled_path, parse_dates=['time'])
    signals = signals.sort_values('time')

    buy_signals = signals[signals['signal'] == 'buy'].copy()
    if len(buy_signals) == 0:
        print(f"{signals_labeled_path} 中没有买入信号")
        return None, None, None, None

    # 确保索引是 datetime
    ohlcv.index = pd.to_datetime(ohlcv.index)
    buy_signals['time'] = pd.to_datetime(buy_signals['time'])

    results = []
    for idx, row in buy_signals.iterrows():
        sig_time = row['time']
        sig_price = row['price']
        d_label = row['D_label']
        xi_label = row['xi_label']

        if sig_time not in ohlcv.index:
            future_idx = ohlcv.index[ohlcv.index >= sig_time]
            if len(future_idx) == 0:
                continue
            sig_time = future_idx[0]

        loc = ohlcv.index.get_loc(sig_time)
        end_loc = min(loc + future_T, len(ohlcv) - 1)
        if loc >= end_loc:
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
    if len(df_results) == 0:
        print(f"{signals_labeled_path} 有效信号数（有足够未来数据）为0")
        return None, None, None, None

    # 分桶
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
    stats = df_results.groupby('D_bucket').agg(
        count=('max_return', 'size'),
        mean_return=('max_return', 'mean'),
        median_return=('max_return', 'median'),
        win_rate=('max_return', lambda x: (x > 0).mean()),
        mean_xi=('xi_label', 'mean')
    ).reset_index()

    bucket_order = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)", "1.0"]
    stats['D_bucket'] = pd.Categorical(stats['D_bucket'], categories=bucket_order, ordered=True)
    stats = stats.sort_values('D_bucket')

    corr, p_value = spearmanr(df_results['D_label'], df_results['max_return'])

    return df_results, stats, corr, p_value


def batch_evaluate(stock_list_path, ohlcv_dir, signals_labeled_dir, future_T=60, output_dir="./eval_results"):
    """
    批量评估多只股票
    stock_list_path: CSV文件，包含股票代码列，例如 'code' 列为 'sh.600000' 或 '600000'
    ohlcv_dir: OHLCV 文件所在目录，文件命名规则为 bar_A_{code_short}_d.csv
    signals_labeled_dir: 标注信号文件所在目录，命名规则为 {code_short}signals_labeled.csv
    output_dir: 输出结果的目录
    """
    os.makedirs(output_dir, exist_ok=True)

    stocks_df = pd.read_csv(stock_list_path, dtype={'code': str})
    all_summaries = []

    for _, row in stocks_df.iterrows():
        code = row['code']  # 例如 'sh.600000'
        # 提取短代码（去掉前缀）
        if '.' in code:
            code_short = code.split('.')[-1]
        else:
            code_short = code

        ohlcv_path = os.path.join(ohlcv_dir, f"bar_A_{code_short}_d.csv")
        signals_path = os.path.join(signals_labeled_dir, f"{code_short}signals_labeled.csv")

        print(f"\n正在评估股票: {code_short}")
        df_res, stats, corr, p = evaluate_labeling(ohlcv_path, signals_path, future_T)

        if stats is None:
            print(f"  跳过 {code_short}，数据不足")
            continue

        # 保存该股票的统计结果到单独文件
        stats.to_csv(os.path.join(output_dir, f"{code_short}_stats.csv"), index=False)

        # 记录汇总信息
        summary = {
            'code': code_short,
            'total_signals': len(df_res),
            'spearman_corr': corr,
            'p_value': p,
            'mean_return_all': df_res['max_return'].mean(),
            'win_rate_all': (df_res['max_return'] > 0).mean(),
            # 也可以记录 D=1.0 桶的表现
            'count_D1': stats.loc[stats['D_bucket'] == '1.0', 'count'].values[0] if '1.0' in stats[
                'D_bucket'].values else 0,
            'mean_return_D1': stats.loc[stats['D_bucket'] == '1.0', 'mean_return'].values[0] if '1.0' in stats[
                'D_bucket'].values else np.nan,
        }
        all_summaries.append(summary)

        # 打印当前股票的结果
        print(f"  有效信号数: {len(df_res)}")
        print(f"  Spearman 相关系数: {corr:.4f} (p={p:.4f})")
        if corr < 0 and p < 0.05:
            print("  ✅ 显著负相关，标注有效")
        elif corr > 0:
            print("  ⚠️ 正相关，标注反向")
        else:
            print("  ⚠️ 无明显相关性")
        print(stats[['D_bucket', 'count', 'mean_return', 'win_rate']].to_string(index=False))

    # 汇总所有股票的结果
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(output_dir, "all_stocks_summary.csv"), index=False)

    # 绘制 Spearman 相关系数分布直方图
    if len(df_summary) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(df_summary['spearman_corr'], bins=20, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Spearman Correlation (D vs future return)')
        plt.ylabel('Number of stocks')
        plt.title('Distribution of Labeling Consistency Across Stocks')
        plt.savefig(os.path.join(output_dir, "spearman_distribution.png"), dpi=150)
        plt.show()

        print("\n===== 所有股票汇总统计 =====")
        print(f"平均相关系数: {df_summary['spearman_corr'].mean():.4f}")
        print(
            f"显著负相关股票数 (corr<0, p<0.05): {((df_summary['spearman_corr'] < 0) & (df_summary['p_value'] < 0.05)).sum()}")
        print(f"正相关股票数: {(df_summary['spearman_corr'] > 0).sum()}")
    else:
        print("没有成功评估任何股票")

    return df_summary


if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    STOCK_LIST = "D://github//RobotMeQ//QuantData//asset_code//a800_stocks_2025.csv"
    OHLCV_DIR = "D:/github/RobotMeQ_Dataset/QuantData/backTest"
    SIGNALS_DIR = "./trade_point_backtest_tea_radical_nature"
    OUTPUT_DIR = "./labeling_eval_batch"

    batch_evaluate(STOCK_LIST, OHLCV_DIR, SIGNALS_DIR, future_T=60, output_dir=OUTPUT_DIR)