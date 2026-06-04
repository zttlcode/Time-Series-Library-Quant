import os
import pandas as pd
import numpy as np
from sktime.datasets import write_dataframe_to_tsfile
from itertools import combinations

import SQData.Asset as SQAsset
import SQData.Indicator as SQIndicator
import SQTool.Tools as SQTools
import SQData.Identify_market_types_helper as IMTHelper

# 你的基础通道：每个计划都保留
# BASE_FEATURES = [
#     'ema10', 'ema20', 'ema60',
#     'macd', 'signal',
#     'adx', 'plus_di', 'minus_di',
#     'atr',
#     'boll_mid', 'boll_upper', 'boll_lower',
#     'rsi', 'obv', 'volume_ma5',
#     'close', 'volume'
# ]
BASE_FEATURES = [
    'ret_5', 'hl_range', 'upper_wick_pct',
    'volume', 'close'
]

# 你要做组合搜索的候选池
DISTANCE_CANDIDATES = [
    'close_ma5_ratio',
    'close_ma20_ratio',
    'close_ma60_ratio',
    'ma5_ma20_gap',
    'ma20_ma60_gap',
    'dist_to_low_20',
    'dist_to_high_20',
    'range_pos_20',
    'dist_to_low_60',
    'dist_to_high_60',
    'range_pos_60',
    'volume_ma20_ratio',
    'volume_zscore_20'
]

# 如果以后还想扫交互特征，也可以单独放池子
INTERACTION_CANDIDATES = [
    'ret_5_x_volume_zscore_20',
    'ret_5_x_dist_to_low_20',
    'close_ma20_x_ret_5',
    'wick_pressure',
    'upper_wick_to_range',
    'lower_wick_to_range',
    'body_to_range'
]

FEATURE_PLAN_SPECS = {}


def register_combo_plans(base_name, base_cols, pool, combo_sizes=(2, 3)):
    """
    自动生成组合计划名 -> 特征列表
    例如：
      feature_tea_radical_nature_dist_v1__2__dist_to_low_20__range_pos_20
    """
    for r in combo_sizes:
        for comb in combinations(pool, r):
            plan_name = f"{base_name}_{r}_" + "_".join(comb)
            FEATURE_PLAN_SPECS[plan_name] = base_cols + list(comb)


# 只注册你想扫的这一类  combo_sizes=(1, )表示基础特征 组合1个特征， (1, 3) 表示组合1个特征、所有3个特征，这个combo_sizes必须是元组
register_combo_plans(
    base_name="ftr",
    base_cols=BASE_FEATURES,
    pool=INTERACTION_CANDIDATES,
    combo_sizes=(2, 3)
)


def _balance_indices_by_label_generic(local_label_list, local_time_list, expected_labels=None, require_all=True):
    """
    对已成功提取出来的有效样本做均衡。
    - local_label_list: 样本标签列表
    - local_time_list:  对应时间列表
    - expected_labels:  期望参与均衡的标签集合，比如 [1,2] 或 [1,2,3,4]
    - require_all:      是否要求 expected_labels 中的每个类都必须存在

    返回：
    - keep_idx: 需要保留的样本索引
    - 若无法均衡，返回 None
    """
    if len(local_label_list) == 0:
        return None

    s = pd.Series(local_label_list)

    # 如果指定了期望标签，只保留这些标签
    if expected_labels is not None:
        s = s[s.isin(expected_labels)]
        if s.empty:
            return None

    counts = s.value_counts()

    # 严格要求所有类别都存在
    if require_all and expected_labels is not None:
        missing = [lab for lab in expected_labels if lab not in counts.index]
        if len(missing) > 0:
            # print(f"⚠️ 缺少类别 {missing}，无法做严格均衡")
            return None

    # 少于2类也无法均衡
    if len(counts) < 2:
        return None

    target_n = int(counts.min())
    if target_n <= 0:
        return None

    keep_idx = []

    # 每个类别都取最后 target_n 条，偏向保留更靠后的时序样本
    for lab in counts.index:
        idx = s[s == lab].index.tolist()
        idx_sorted = sorted(idx, key=lambda i: local_time_list[i])
        keep_idx.extend(idx_sorted[-target_n:])

    # 最终按时间排序
    keep_idx = sorted(set(keep_idx), key=lambda i: local_time_list[i])

    # 再保险检查一次：每类是否真的相等
    final_labels = [local_label_list[i] for i in keep_idx]
    final_counts = pd.Series(final_labels).value_counts()
    if final_counts.nunique() != 1:
        return None

    return keep_idx


def tea_radical_nature_handling_uneven_samples1(labeled):
    # 统计每个 label 的数量
    label_counts = labeled['label'].value_counts()
    min_label_count = label_counts.min()

    # 创建一个空的 DataFrame 来存储处理后的数据
    final_data = []

    # 按照连续相同的 label 分组
    labeled['group'] = (labeled['label'] != labeled['label'].shift()).cumsum()

    # 对每个 label 类型进行处理
    for label in label_counts.index:
        label_data = labeled[labeled['label'] == label]

        # 计算该 label 所有组的数量
        group_count = label_data['group'].nunique()

        # 如果 group_count 很大，确保每个组至少保留一行
        rows_per_group = min_label_count // group_count
        if rows_per_group == 0:
            rows_per_group = 1  # 如果计算结果为0，则至少保留1行每组

        # 初始化存储裁剪后的行
        cropped_data = []

        # 获取所有的分组
        groups = list(label_data.groupby('group'))

        # 累计已经保留的数据量
        accumulated_data = 0

        for idx, (group, group_data) in enumerate(groups):
            group_size = len(group_data)

            # len(groups) 是总的分组数，idx 是当前遍历的索引，所以 len(groups) - idx - 1 就是剩余的次数。
            remaining = len(groups) - idx - 1

            if group_count < min_label_count:
                # 如果已经裁剪的数据量和剩余数据量合起来超过了 min_label_count，停止迭代
                if accumulated_data + remaining <= min_label_count:
                    cropped_data.append(group_data)
                    accumulated_data += len(group_data)
                    continue

            # 如果组的大小超过每组应保留的行数，则裁掉前面部分
            if group_size > rows_per_group:
                group_data = group_data.tail(rows_per_group)

            # 将裁剪后的组数据添加到 cropped_data
            cropped_data.append(group_data)
            accumulated_data += len(group_data)

        # 将裁剪后的数据合并成一个 DataFrame
        cropped_data = pd.concat(cropped_data)

        # 如果裁剪后的数据总行数超过 min_label_count，裁掉前面的多余行
        if len(cropped_data) > min_label_count:
            cropped_data = cropped_data.tail(min_label_count)

        # 将裁剪后的数据添加到 final_data
        final_data.append(cropped_data)

    if not final_data:
        return pd.DataFrame()
    # 将 final_data 合并成一个 DataFrame
    final_data = pd.concat(final_data)

    # 对最终数据进行排序，保持时间顺序
    final_data = final_data.sort_index()

    # 输出结果
    final_data = final_data.drop(columns=['group'])  # 删除辅助列
    return final_data


def fuzzy_nature_handling_uneven_samples1(labeled):
    """ 平衡 label 使其各类别数量相等，并均匀删除 """
    df = labeled

    # 统计 label 数量
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()  # 找到最少的 label 数

    # 确保 1-3 和 2-4 交易对完整
    group_13 = df[df['label'].isin([1, 3])]
    group_24 = df[df['label'].isin([2, 4])]

    # 计算需要删除的行数
    excess_13 = len(group_13) // 2 - min_count
    excess_24 = len(group_24) // 2 - min_count
    if len(group_13) == 0:
        return pd.DataFrame()

    def remove_evenly(group, excess_count):
        """ 在整个 group 中均匀删除成对数据 """
        if excess_count <= 0:
            return group  # 无需删除
        total_pairs = len(group) // 2  # 计算交易对数量
        step = total_pairs / excess_count  # 计算均匀删除步长
        remove_indices = np.round(np.arange(0, total_pairs, step)).astype(int)  # 均匀选择要删除的索引
        mask = np.ones(len(group), dtype=bool)
        for idx in remove_indices:
            mask[idx * 2: idx * 2 + 2] = False  # 确保成对删除
        return group[mask]

    # 均匀删除多余的 1-3 和 2-4
    balanced_13 = remove_evenly(group_13, excess_13)
    balanced_24 = remove_evenly(group_24, excess_24)

    # 合并数据并按时间排序
    balanced_df = pd.concat([balanced_13, balanced_24]).sort_values(by="time")

    def is_valid_trade_sequence(dfv):
        """检查数据是否符合交易对 buy-sell 规则"""
        if dfv.iloc[0]["signal"] != "buy" or dfv.iloc[-1]["signal"] != "sell":
            print("数据不符合 buy 开头、sell 结尾的规则！")
            return False

        # 检查相邻交易对
        signals = dfv["signal"].values
        for i in range(0, len(signals) - 1, 2):  # 2步长遍历
            if signals[i] != "buy" or signals[i + 1] != "sell":
                print(f"数据不符合交易对规则，在索引 {i}-{i + 1} 处发现异常！")
                return False
        return True

    # **增加交易对校验**
    if not is_valid_trade_sequence(balanced_df):
        print(f"错误！交易对结构被破坏，请检查算法！")
        return

    return balanced_df


def tea_radical_nature_point_to_ts3(assetList, temp_data_dict, temp_label_list, time_point_step, handle_uneven_samples,
                                    strategy_name, label_name, feature_plan_name):
    # 加载数据
    item = 'trade_point_backtest_' + strategy_name
    concat_labeled_filePath = (SQTools.read_config("SQData", item)
                               + assetList[0].assetsMarket
                               + "_"
                               + assetList[0].assetsCode + str(label_name) + ".csv")
    index_d_filepath = (SQTools.read_config("SQData", "backtest_bar")
                        + "bar_"
                        + assetList[0].assetsMarket
                        + "_"
                        + "000001_index_d" + ".csv")
    data_d_filePath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[0].assetsMarket
                       + "_"
                       + assetList[0].assetsCode
                       + '_d.csv')
    data_60_df_filePath = (SQTools.read_config("SQData", "backtest_bar")
                           + "bar_"
                           + assetList[0].assetsMarket
                           + "_"
                           + assetList[0].assetsCode
                           + '_60.csv')
    if not os.path.exists(concat_labeled_filePath):
        return None
    concat_labeled = pd.read_csv(concat_labeled_filePath, index_col="time", parse_dates=True)
    index_d = pd.read_csv(index_d_filepath, index_col="date", parse_dates=True)
    data_d = pd.read_csv(data_d_filePath, index_col="time", parse_dates=True)
    data_60 = pd.read_csv(data_60_df_filePath, index_col="time", parse_dates=True)

    # 是否处理样本不均
    if handle_uneven_samples:
        handled_uneven_filepath = (SQTools.read_config("SQData", item)
                                   + assetList[0].assetsMarket
                                   + "_"
                                   + assetList[0].assetsCode + str(label_name)
                                   + "_handled_uneven" + ".csv")
        if not os.path.exists(handled_uneven_filepath):
            concat_labeled = tea_radical_nature_handling_uneven_samples1(concat_labeled)
            if concat_labeled.empty:
                print(assetList[0].assetsCode, "处理样本不均终止")
                return None
            concat_labeled.to_csv(handled_uneven_filepath, index=True)
        else:
            concat_labeled = pd.read_csv(handled_uneven_filepath, index_col="time", parse_dates=True)
            # print(assetList[0].assetsCode, "样本", concat_labeled['label'].value_counts())

    # 遍历 concat_labeled 数据
    for labeled_time, labeled_row in concat_labeled.iterrows():
        labeled_date = labeled_time.date()
        labeled_hour = labeled_time.hour
        # 在 backtest_bar 中寻找同一日的数据
        if pd.Timestamp(labeled_date) in index_d.index:
            index_d_row_index = index_d.index.get_loc(pd.Timestamp(labeled_date))
            if index_d_row_index >= time_point_step:
                if feature_plan_name == 'feature2':
                    data_0_tmp = index_d.iloc[index_d_row_index - time_point_step: index_d_row_index].reset_index(
                        drop=True)
                    data_0_tmp = SQIndicator.calMACD(data_0_tmp)
                    data_0_tmp = SQIndicator.calKDJ(data_0_tmp)

                    MACD_0 = data_0_tmp["MACD"]
                    DIF_0 = data_0_tmp["DIF"]
                    DEA_0 = data_0_tmp["DEA"]
                    K_0 = data_0_tmp["K"]
                    D_0 = data_0_tmp["D"]
                    J_0 = data_0_tmp["J"]
                    close_0 = data_0_tmp["close"]

                    if (MACD_0.isna().any() or DIF_0.isna().any() or DEA_0.isna().any() or K_0.isna().any()
                            or D_0.isna().any() or J_0.isna().any() or close_0.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    index_d_close = index_d.iloc[index_d_row_index - time_point_step: index_d_row_index][
                        "close"].reset_index(drop=True)
                    index_d_volume = index_d.iloc[index_d_row_index - time_point_step: index_d_row_index][
                        "volume"].reset_index(drop=True)
                    if index_d_close.isna().any() or index_d_volume.isna().any():
                        continue  # 数据NaN，跳过
            else:
                continue  # backtest_bar 越界，跳过
        else:
            continue  # 无匹配日期，跳过

        # 在 d.csv 中寻找同一日的数据
        if pd.Timestamp(labeled_date) in data_d.index:
            d_row_index = data_d.index.get_loc(pd.Timestamp(labeled_date))
            if d_row_index >= time_point_step:
                if feature_plan_name == 'feature2':
                    data_1_tmp = data_d.iloc[d_row_index - time_point_step: d_row_index].reset_index(
                        drop=True)
                    data_1_tmp = SQIndicator.calMACD(data_1_tmp)
                    data_1_tmp = SQIndicator.calKDJ(data_1_tmp)

                    MACD_1 = data_1_tmp["MACD"]
                    DIF_1 = data_1_tmp["DIF"]
                    DEA_1 = data_1_tmp["DEA"]
                    K_1 = data_1_tmp["K"]
                    D_1 = data_1_tmp["D"]
                    J_1 = data_1_tmp["J"]
                    close_1 = data_1_tmp["close"]

                    if (MACD_1.isna().any() or DIF_1.isna().any() or DEA_1.isna().any() or K_1.isna().any()
                            or D_1.isna().any() or J_1.isna().any() or close_1.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    d_close = data_d.iloc[d_row_index - time_point_step: d_row_index]["close"].reset_index(drop=True)
                    d_volume = data_d.iloc[d_row_index - time_point_step: d_row_index]["volume"].reset_index(drop=True)
                    if d_close.isna().any() or d_volume.isna().any():
                        continue  # 数据NaN，跳过
            else:
                continue  # d.csv 越界，跳过
        else:
            continue  # 无匹配日期，跳过

        # 在 60.csv 中寻找同一日且同一小时的数据
        if labeled_hour == 9 or labeled_hour == 13:
            # 这俩匹配不上，只能改一下时间
            labeled_hour += 1
        day_hour_filter = (data_60.index.date == labeled_date) & (data_60.index.hour == labeled_hour)
        matched_60 = data_60[day_hour_filter]
        if len(matched_60) > 0:
            matched_60_index = matched_60.index[-1]
            matched_60_row_index = data_60.index.get_loc(matched_60_index)
            if matched_60_row_index >= time_point_step:
                if feature_plan_name == 'feature2':
                    data_2_tmp = data_60.iloc[matched_60_row_index - time_point_step: matched_60_row_index].reset_index(
                        drop=True)
                    data_2_tmp = SQIndicator.calMACD(data_2_tmp)
                    data_2_tmp = SQIndicator.calKDJ(data_2_tmp)

                    MACD_2 = data_2_tmp["MACD"]
                    DIF_2 = data_2_tmp["DIF"]
                    DEA_2 = data_2_tmp["DEA"]
                    K_2 = data_2_tmp["K"]
                    D_2 = data_2_tmp["D"]
                    J_2 = data_2_tmp["J"]
                    close_2 = data_2_tmp["close"]

                    if (MACD_2.isna().any() or DIF_2.isna().any() or DEA_2.isna().any() or K_2.isna().any()
                            or D_2.isna().any() or J_2.isna().any() or close_2.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    close_60 = data_60.iloc[matched_60_row_index - time_point_step: matched_60_row_index][
                        "close"].reset_index(drop=True)
                    volume_60 = data_60.iloc[matched_60_row_index - time_point_step: matched_60_row_index][
                        "volume"].reset_index(drop=True)
                    if close_60.isna().any() or volume_60.isna().any():
                        continue  # 数据NaN，跳过
            else:
                continue  # 60.csv 越界，跳过
        else:
            continue  # 无匹配日期或小时，跳过

        # # 如果通过所有越界检查，将数据存入字典  标签存入列表
        if feature_plan_name == 'feature2':
            temp_data_dict['MACD_0'].append(MACD_0)
            temp_data_dict['DIF_0'].append(DIF_0)
            temp_data_dict['DEA_0'].append(DEA_0)
            temp_data_dict['K_0'].append(K_0)
            temp_data_dict['D_0'].append(D_0)
            temp_data_dict['J_0'].append(J_0)
            temp_data_dict['close_0'].append(close_0)
            temp_data_dict['MACD_1'].append(MACD_1)
            temp_data_dict['DIF_1'].append(DIF_1)
            temp_data_dict['DEA_1'].append(DEA_1)
            temp_data_dict['K_1'].append(K_1)
            temp_data_dict['D_1'].append(D_1)
            temp_data_dict['J_1'].append(J_1)
            temp_data_dict['close_1'].append(close_1)
            temp_data_dict['MACD_2'].append(MACD_2)
            temp_data_dict['DIF_2'].append(DIF_2)
            temp_data_dict['DEA_2'].append(DEA_2)
            temp_data_dict['K_2'].append(K_2)
            temp_data_dict['D_2'].append(D_2)
            temp_data_dict['J_2'].append(J_2)
            temp_data_dict['close_2'].append(close_2)
        else:
            temp_data_dict['index_d_close'].append(index_d_close)
            temp_data_dict['index_d_volume'].append(index_d_volume)
            temp_data_dict['d_close'].append(d_close)
            temp_data_dict['d_volume'].append(d_volume)
            temp_data_dict['close_60'].append(close_60)
            temp_data_dict['volume_60'].append(volume_60)

        temp_label_list.append(labeled_row['label'])

    print(assetList[0].assetsCode, "结束", len(temp_label_list))
    return "success"


def tea_radical_nature_point_to_ts2(assetList, temp_data_dict, temp_label_list, time_point_step, handle_uneven_samples,
                                    strategy_name, label_name, feature_plan_name):
    # 加载数据
    item = 'trade_point_backtest_' + strategy_name
    labeled_filePath = (SQTools.read_config("SQData", item)
                        + assetList[0].assetsMarket
                        + "_"
                        + assetList[0].assetsCode
                        + "_"
                        + assetList[0].barEntity.timeLevel
                        + str(label_name)
                        + ".csv")
    if not os.path.exists(labeled_filePath):
        return None
    data_0_filepath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[0].assetsMarket
                       + "_"
                       + assetList[0].assetsCode
                       + "_"
                       + assetList[0].barEntity.timeLevel
                       + ".csv")
    data_1_filePath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[1].assetsMarket
                       + "_"
                       + assetList[1].assetsCode
                       + "_"
                       + assetList[1].barEntity.timeLevel
                       + ".csv")
    data_2_filePath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[2].assetsMarket
                       + "_"
                       + assetList[2].assetsCode
                       + "_"
                       + assetList[2].barEntity.timeLevel
                       + ".csv")

    labeled = pd.read_csv(labeled_filePath, index_col="time", parse_dates=True)
    data_0 = pd.read_csv(data_0_filepath, index_col="time", parse_dates=True)
    #data_1 = pd.read_csv(data_1_filePath, index_col="time", parse_dates=True)
    #data_2 = pd.read_csv(data_2_filePath, index_col="time", parse_dates=True)
    data_1 = pd.read_csv(data_1_filePath, parse_dates=["time"])
    data_2 = pd.read_csv(data_2_filePath, parse_dates=["time"])

    # 是否处理样本不均
    if handle_uneven_samples:
        handled_uneven_filepath = (SQTools.read_config("SQData", item)
                                   + assetList[0].assetsMarket
                                   + "_"
                                   + assetList[0].assetsCode
                                   + "_"
                                   + assetList[0].barEntity.timeLevel
                                   + str(label_name)
                                   + "_handled_uneven" + ".csv")
        if not os.path.exists(handled_uneven_filepath):
            labeled = tea_radical_nature_handling_uneven_samples1(labeled)
            if labeled.empty:
                print(assetList[0].assetsCode, "处理样本不均终止")
                return None
            labeled.to_csv(handled_uneven_filepath, index=True)
        else:
            labeled = pd.read_csv(handled_uneven_filepath, index_col="time", parse_dates=True)
        # print(assetList[0].assetsCode, "样本", concat_labeled['label'].value_counts())

    # 遍历 concat_labeled 数据
    for labeled_time, labeled_row in labeled.iterrows():
        """1、寻找本级别匹配行 此为5分钟级别"""
        data_0_filter = (data_0.index == labeled_time)
        matched_0 = data_0[data_0_filter]
        if len(matched_0) > 0:
            matched_0_index = matched_0.index[-1]
            matched_0_row_index = data_0.index.get_loc(matched_0_index)
            if matched_0_row_index >= time_point_step:
                # 时间索引用完了，跟tea_radical_nature_label3一样删掉它，不然计算指标会报错
                data_0_tmp = data_0.iloc[matched_0_row_index - time_point_step: matched_0_row_index].reset_index(
                    drop=True)
                data_0_tmp = SQIndicator.calMACD(data_0_tmp)
                data_0_tmp = SQIndicator.calKDJ(data_0_tmp)

                MACD_0 = data_0_tmp["MACD"]
                DIF_0 = data_0_tmp["DIF"]
                DEA_0 = data_0_tmp["DEA"]
                K_0 = data_0_tmp["K"]
                D_0 = data_0_tmp["D"]
                J_0 = data_0_tmp["J"]
                close_0 = data_0_tmp["close"]

                if (MACD_0.isna().any() or DIF_0.isna().any() or DEA_0.isna().any() or K_0.isna().any()
                        or D_0.isna().any() or J_0.isna().any() or close_0.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # backtest_bar 越界，跳过
        else:
            continue  # 无匹配日期，跳过
        """2、寻找上级匹配行 计算15分钟K线的区间"""
        data_1_start_time = labeled_time.floor("15T")  # 向下取整到最近的15分钟
        data_1_end_time = data_1_start_time + pd.Timedelta(minutes=15)
        # 在 15分钟级别 中查找符合区间的行
        data_1_mask = (data_1["time"] > data_1_start_time) & (data_1["time"] <= data_1_end_time)
        data_1_matching_indices = data_1.index[data_1_mask].to_numpy()  # 使用 to_numpy() 解决 FutureWarning

        if data_1_matching_indices.size > 0:
            data_1_prev_index = data_1_matching_indices[-1] - 1  # 获取上一行的索引

            # 检查是否越界
            if data_1_prev_index < 0:
                continue

            # 获取上一行的time
            # data_1_prev_row_time = data_1.loc[data_1_prev_index, "time"]
            # data_1_row_index = data_1.index.get_loc(pd.Timestamp(data_1_prev_row_time))
            data_1_row_index = data_1_prev_index

            if data_1_row_index >= time_point_step:
                data_1_tmp = data_1.iloc[data_1_row_index - time_point_step: data_1_row_index].reset_index(drop=True)
                data_1_tmp = SQIndicator.calMACD(data_1_tmp)
                data_1_tmp = SQIndicator.calKDJ(data_1_tmp)

                MACD_1 = data_1_tmp["MACD"]
                DIF_1 = data_1_tmp["DIF"]
                DEA_1 = data_1_tmp["DEA"]
                K_1 = data_1_tmp["K"]
                D_1 = data_1_tmp["D"]
                J_1 = data_1_tmp["J"]
                close_1 = data_1_tmp["close"]

                if (MACD_1.isna().any() or DIF_1.isna().any() or DEA_1.isna().any() or K_1.isna().any()
                        or D_1.isna().any() or J_1.isna().any() or close_1.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # d.csv 越界，跳过
        else:
            continue  # 无匹配日期，跳过

        """3、寻找上级匹配行 计算30分钟K线的区间"""
        data_2_start_time = labeled_time.floor("30T")  # 向下取整到最近的15分钟
        data_2_end_time = data_2_start_time + pd.Timedelta(minutes=30)
        # 在 30分钟级别 中查找符合区间的行
        data_2_mask = (data_2["time"] > data_2_start_time) & (data_2["time"] <= data_2_end_time)
        data_2_matching_indices = data_2.index[data_2_mask].to_numpy()  # 使用 to_numpy() 解决 FutureWarning

        if data_2_matching_indices.size > 0:
            data_2_prev_index = data_2_matching_indices[-1] - 1  # 获取上一行的索引

            # 检查是否越界
            if data_2_prev_index < 0:
                continue

            # 获取上一行的time
            # data_2_prev_row_time = data_1.loc[data_2_prev_index, "time"]
            # data_2_row_index = data_1.index.get_loc(pd.Timestamp(data_2_prev_row_time))
            data_2_row_index = data_2_prev_index

            if data_2_row_index >= time_point_step:
                data_2_tmp = data_2.iloc[data_2_row_index - time_point_step: data_2_row_index].reset_index(drop=True)
                data_2_tmp = SQIndicator.calMACD(data_2_tmp)
                data_2_tmp = SQIndicator.calKDJ(data_2_tmp)

                MACD_2 = data_2_tmp["MACD"]
                DIF_2 = data_2_tmp["DIF"]
                DEA_2 = data_2_tmp["DEA"]
                K_2 = data_2_tmp["K"]
                D_2 = data_2_tmp["D"]
                J_2 = data_2_tmp["J"]
                close_2 = data_2_tmp["close"]

                if (MACD_2.isna().any() or DIF_2.isna().any() or DEA_2.isna().any() or K_2.isna().any()
                        or D_2.isna().any() or J_2.isna().any() or close_2.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # 60.csv 越界，跳过
        else:
            continue  # 无匹配日期或小时，跳过

        # # 如果通过所有越界检查，将数据存入字典  标签存入列表
        temp_data_dict['MACD_0'].append(MACD_0)
        temp_data_dict['DIF_0'].append(DIF_0)
        temp_data_dict['DEA_0'].append(DEA_0)
        temp_data_dict['K_0'].append(K_0)
        temp_data_dict['D_0'].append(D_0)
        temp_data_dict['J_0'].append(J_0)
        temp_data_dict['close_0'].append(close_0)
        temp_data_dict['MACD_1'].append(MACD_1)
        temp_data_dict['DIF_1'].append(DIF_1)
        temp_data_dict['DEA_1'].append(DEA_1)
        temp_data_dict['K_1'].append(K_1)
        temp_data_dict['D_1'].append(D_1)
        temp_data_dict['J_1'].append(J_1)
        temp_data_dict['close_1'].append(close_1)
        temp_data_dict['MACD_2'].append(MACD_2)
        temp_data_dict['DIF_2'].append(DIF_2)
        temp_data_dict['DEA_2'].append(DEA_2)
        temp_data_dict['K_2'].append(K_2)
        temp_data_dict['D_2'].append(D_2)
        temp_data_dict['J_2'].append(J_2)
        temp_data_dict['close_2'].append(close_2)

        temp_label_list.append(labeled_row['label'])

    print(assetList[0].assetsCode, "结束", len(temp_label_list))
    return "success"


def fuzzy_nature_point_to_ts2(assetList, temp_data_dict, temp_label_list, time_point_step, handle_uneven_samples,
                              strategy_name, label_name, feature_plan_name):
    # 加载数据
    item = 'trade_point_backtest_' + strategy_name
    labeled_filePath = (SQTools.read_config("SQData", item)
                        + assetList[2].assetsMarket
                        + "_"
                        + assetList[2].assetsCode
                        + "_"
                        + assetList[2].barEntity.timeLevel
                        + str(label_name)
                        + ".csv")

    if not os.path.exists(labeled_filePath):
        return None

    data_0_filepath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[2].assetsMarket
                       + "_"
                       + assetList[2].assetsCode
                       + "_"
                       + assetList[2].barEntity.timeLevel
                       + ".csv")
    data_1_filePath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[3].assetsMarket
                       + "_"
                       + assetList[3].assetsCode
                       + "_"
                       + assetList[3].barEntity.timeLevel
                       + ".csv")
    data_2_filePath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[4].assetsMarket
                       + "_"
                       + assetList[4].assetsCode
                       + "_"
                       + assetList[4].barEntity.timeLevel
                       + ".csv")

    labeled = pd.read_csv(labeled_filePath, index_col="time", parse_dates=True)
    data_0 = pd.read_csv(data_0_filepath, index_col="time", parse_dates=True)
    data_1 = pd.read_csv(data_1_filePath, index_col="time", parse_dates=True)
    data_2 = pd.read_csv(data_2_filePath, index_col="time", parse_dates=True)

    # 是否处理样本不均
    if handle_uneven_samples:
        handled_uneven_filepath = (SQTools.read_config("SQData", item)
                                   + assetList[2].assetsMarket
                                   + "_"
                                   + assetList[2].assetsCode
                                   + "_"
                                   + assetList[2].barEntity.timeLevel
                                   + str(label_name)
                                   + "_handled_uneven" + ".csv")
        if not os.path.exists(handled_uneven_filepath):
            labeled = fuzzy_nature_handling_uneven_samples1(labeled)
            if labeled.empty:
                print(assetList[0].assetsCode, "处理样本不均终止")
                return None
            labeled.to_csv(handled_uneven_filepath, index=True)
        else:
            labeled = pd.read_csv(handled_uneven_filepath, index_col="time", parse_dates=True)

    # 遍历 labeled 数据
    for labeled_time, labeled_row in labeled.iterrows():
        """1、寻找本级别匹配行 此为30分钟级别"""
        data_0_filter = (data_0.index == labeled_time)
        matched_0 = data_0[data_0_filter]
        if len(matched_0) > 0:
            matched_0_index = matched_0.index[-1]
            matched_0_row_index = data_0.index.get_loc(matched_0_index)
            if matched_0_row_index >= (time_point_step + 60):
                # 时间索引用完了，跟tea_radical_nature_label3一样删掉它，不然计算指标会报错
                data_0_tmp = data_0.iloc[matched_0_row_index - (time_point_step + 60): matched_0_row_index].reset_index(
                    drop=True)
                data_0_tmp = SQIndicator.calMACD(data_0_tmp)
                data_0_tmp = SQIndicator.calMA(data_0_tmp)

                MACD_0 = data_0_tmp["MACD"][59:]
                DIF_0 = data_0_tmp["DIF"][59:]
                MA_5_0 = data_0_tmp["MA_5"][59:]
                MA_10_0 = data_0_tmp["MA_10"][59:]
                MA_60_0 = data_0_tmp["MA_60"][59:]
                close_0 = data_0_tmp["close"][59:]

                if (MACD_0.isna().any() or DIF_0.isna().any() or MA_5_0.isna().any()
                        or MA_10_0.isna().any() or MA_60_0.isna().any() or close_0.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # backtest_bar 越界，跳过
        else:
            continue  # 无匹配日期，跳过

        """2、寻找上级匹配行 在 60.csv 中寻找同一日且同一小时的数据"""
        labeled_date = labeled_time.date()
        labeled_hour = labeled_time.hour

        if labeled_hour == 9 or labeled_hour == 13:
            labeled_hour += 1  # 这俩匹配不上，只能改一下时间
        day_hour_filter = (data_1.index.date == labeled_date) & (data_1.index.hour == labeled_hour)
        matched_60 = data_1[day_hour_filter]
        if len(matched_60) > 0:
            matched_60_index = matched_60.index[-1]
            data_1_row_index = data_1.index.get_loc(matched_60_index)
            if data_1_row_index >= (time_point_step + 60):
                data_1_tmp = data_1.iloc[data_1_row_index - (time_point_step + 60): data_1_row_index].reset_index(
                    drop=True)
                data_1_tmp = SQIndicator.calMACD(data_1_tmp)
                data_1_tmp = SQIndicator.calMA(data_1_tmp)

                MACD_1 = data_1_tmp["MACD"][59:]
                DIF_1 = data_1_tmp["DIF"][59:]
                MA_5_1 = data_1_tmp["MA_5"][59:]
                MA_10_1 = data_1_tmp["MA_10"][59:]
                MA_60_1 = data_1_tmp["MA_60"][59:]
                close_1 = data_1_tmp["close"][59:]

                if (MACD_1.isna().any() or DIF_1.isna().any() or MA_5_1.isna().any()
                        or MA_10_1.isna().any() or MA_60_1.isna().any() or close_1.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # d.csv 越界，跳过
        else:
            continue  # 无匹配日期，跳过

        """3、寻找上级匹配行 在 d.csv 中寻找同一日的数据"""
        if pd.Timestamp(labeled_date) in data_2.index:
            data_2_row_index = data_2.index.get_loc(pd.Timestamp(labeled_date))
            if data_2_row_index >= (time_point_step + 60):
                data_2_tmp = data_2.iloc[data_2_row_index - (time_point_step + 60): data_2_row_index].reset_index(
                    drop=True)
                data_2_tmp = SQIndicator.calMACD(data_2_tmp)
                data_2_tmp = SQIndicator.calMA(data_2_tmp)

                MACD_2 = data_2_tmp["MACD"][59:]
                DIF_2 = data_2_tmp["DIF"][59:]
                MA_5_2 = data_2_tmp["MA_5"][59:]
                MA_10_2 = data_2_tmp["MA_10"][59:]
                MA_60_2 = data_2_tmp["MA_60"][59:]
                close_2 = data_2_tmp["close"][59:]

                if (MACD_2.isna().any() or DIF_2.isna().any() or MA_5_2.isna().any()
                        or MA_10_2.isna().any() or MA_60_2.isna().any() or close_2.isna().any()):
                    continue  # 数据NaN，跳过
            else:
                continue  # 60.csv 越界，跳过
        else:
            continue  # 无匹配日期或小时，跳过

        # # 如果通过所有越界检查，将数据存入字典  标签存入列表
        temp_data_dict['MACD_0'].append(MACD_0)
        temp_data_dict['DIF_0'].append(DIF_0)
        temp_data_dict['MA_5_0'].append(MA_5_0)
        temp_data_dict['MA_10_0'].append(MA_10_0)
        temp_data_dict['MA_60_0'].append(MA_60_0)
        temp_data_dict['close_0'].append(close_0)
        temp_data_dict['MACD_1'].append(MACD_1)
        temp_data_dict['DIF_1'].append(DIF_1)
        temp_data_dict['MA_5_1'].append(MA_5_1)
        temp_data_dict['MA_10_1'].append(MA_10_1)
        temp_data_dict['MA_60_1'].append(MA_60_1)
        temp_data_dict['close_1'].append(close_1)
        temp_data_dict['MACD_2'].append(MACD_2)
        temp_data_dict['DIF_2'].append(DIF_2)
        temp_data_dict['MA_5_2'].append(MA_5_2)
        temp_data_dict['MA_10_2'].append(MA_10_2)
        temp_data_dict['MA_60_2'].append(MA_60_2)
        temp_data_dict['close_2'].append(close_2)

        temp_label_list.append(labeled_row['label'])

    print(assetList[0].assetsCode, "结束", len(temp_label_list))
    return "success"


def build_feature_bank_tea_radical_nature(data_0: pd.DataFrame) -> pd.DataFrame:
    df = data_0.copy()

    # 先算你原来 feature_all 里用的基础指标
    df = IMTHelper.calculate_indicators(df)

    # 我自己找的基础
    df['ret_5'] = df['close'].pct_change(5)
    df['hl_range'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close'].replace(0, np.nan)

    eps = 1e-12

    # ===== 距离型特征 =====
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    df['close_ma5_ratio'] = df['close'] / df['ma5'].replace(0, np.nan) - 1.0
    df['close_ma20_ratio'] = df['close'] / df['ma20'].replace(0, np.nan) - 1.0
    df['close_ma60_ratio'] = df['close'] / df['ma60'].replace(0, np.nan) - 1.0

    df['ma5_ma20_gap'] = df['ma5'] / df['ma20'].replace(0, np.nan) - 1.0
    df['ma20_ma60_gap'] = df['ma20'] / df['ma60'].replace(0, np.nan) - 1.0

    df['low_20'] = df['low'].rolling(20).min()
    df['high_20'] = df['high'].rolling(20).max()
    df['low_60'] = df['low'].rolling(60).min()
    df['high_60'] = df['high'].rolling(60).max()

    df['dist_to_low_20'] = df['close'] / df['low_20'].replace(0, np.nan) - 1.0
    df['dist_to_high_20'] = df['close'] / df['high_20'].replace(0, np.nan) - 1.0
    df['range_pos_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, np.nan)

    df['dist_to_low_60'] = df['close'] / df['low_60'].replace(0, np.nan) - 1.0
    df['dist_to_high_60'] = df['close'] / df['high_60'].replace(0, np.nan) - 1.0
    df['range_pos_60'] = (df['close'] - df['low_60']) / (df['high_60'] - df['low_60']).replace(0, np.nan)

    vol_ma20 = df['volume'].rolling(20).mean()
    vol_std20 = df['volume'].rolling(20).std(ddof=0)
    df['volume_ma20_ratio'] = df['volume'] / vol_ma20.replace(0, np.nan) - 1.0
    df['volume_zscore_20'] = (df['volume'] - vol_ma20) / vol_std20.replace(0, np.nan)

    # ===== 交互型特征 =====
    candle_range = (df['high'] - df['low']).replace(0, np.nan)
    df['body_to_range'] = (df['close'] - df['open']) / candle_range
    df['upper_wick_to_range'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
    df['lower_wick_to_range'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
    df['wick_pressure'] = df['lower_wick_to_range'] - df['upper_wick_to_range']

    df['ret_5_x_volume_zscore_20'] = df['ret_5'] * df['volume_zscore_20'] if 'ret_5' in df.columns else np.nan
    df['ret_5_x_dist_to_low_20'] = df['ret_5'] * df['dist_to_low_20'] if 'ret_5' in df.columns else np.nan
    df['close_ma20_x_ret_5'] = df['close_ma20_ratio'] * df['ret_5'] if 'ret_5' in df.columns else np.nan

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def single_time_level_point_to_ts(assetList, temp_data_dict, temp_label_list, time_point_step, handle_uneven_samples,
                                  strategy_name, label_name, feature_plan_name,
                                  classification, classification_direction):
    # 加载数据
    if strategy_name == 'identify_Market_Types':
        item = 'market_condition_backtest'
    else:
        item = 'trade_point_backtest_' + strategy_name
    labeled_filePath = (SQTools.read_config("SQData", item)
                        + assetList[0].assetsMarket
                        + "_"
                        + assetList[0].assetsCode
                        + "_"
                        + assetList[0].barEntity.timeLevel
                        + str(label_name)
                        + ".csv")
    data_0_filepath = (SQTools.read_config("SQData", "backtest_bar")
                       + "bar_"
                       + assetList[0].assetsMarket
                       + "_"
                       + assetList[0].assetsCode
                       + "_"
                       + assetList[0].barEntity.timeLevel
                       + ".csv")
    if not os.path.exists(labeled_filePath):
        return None
    labeled = pd.read_csv(labeled_filePath, index_col="time", parse_dates=True)
    data_0 = pd.read_csv(data_0_filepath, index_col="time", parse_dates=True)

    # 计算所有指标
    if feature_plan_name == 'feature_all':
        data_0 = IMTHelper.calculate_indicators(data_0)
    elif feature_plan_name == 'feature_tea_radical_nature':
        data_0['ret_5'] = data_0['close'].pct_change(5)
        data_0['hl_range'] = (data_0['high'] - data_0['low']) / data_0['close'].replace(0, np.nan)
        data_0['upper_wick_pct'] = (data_0['high'] - data_0[['open', 'close']].max(axis=1)) / data_0[
            'close'].replace(0, np.nan)
        # 先拿基础列
        close = data_0['close']
        high = data_0['high']
        low = data_0['low']

        # # ===== 距离型：最近 20 / 60 根区间的位置 =====
        data_0['low_20'] = low.rolling(20).min()
        data_0['high_20'] = high.rolling(20).max()

        data_0['dist_to_low_20'] = close / data_0['low_20'].replace(0, np.nan) - 1.0
        data_0['dist_to_high_20'] = close / data_0['high_20'].replace(0, np.nan) - 1.0
        data_0['range_pos_20'] = (close - data_0['low_20']) / (
                data_0['high_20'] - data_0['low_20']
        ).replace(0, np.nan)

        # 清理极值
        data_0.replace([np.inf, -np.inf], np.nan, inplace=True)

    elif feature_plan_name == 'feature_basic_plus':
        data_0['ret_5'] = data_0['close'].pct_change(5)

        data_0['hl_range'] = (data_0['high'] - data_0['low']) / data_0['close'].replace(0, np.nan)
        data_0['upper_wick_pct'] = (data_0['high'] - data_0[['open', 'close']].max(axis=1)) / data_0[
            'close'].replace(0, np.nan)
        # 先拿基础列
        close = data_0['close']
        open_ = data_0['open']
        high = data_0['high']
        low = data_0['low']
        volume = data_0['volume']

        # # ===== 距离型：最近 20 / 60 根区间的位置 =====
        data_0['low_20'] = low.rolling(20).min()
        data_0['high_20'] = high.rolling(20).max()

        data_0['dist_to_low_20'] = close / data_0['low_20'].replace(0, np.nan) - 1.0
        data_0['dist_to_high_20'] = close / data_0['high_20'].replace(0, np.nan) - 1.0
        data_0['range_pos_20'] = (close - data_0['low_20']) / (
                data_0['high_20'] - data_0['low_20']
        ).replace(0, np.nan)
        # 清理极值
        data_0.replace([np.inf, -np.inf], np.nan, inplace=True)
    elif feature_plan_name in FEATURE_PLAN_SPECS:
        data_0 = build_feature_bank_tea_radical_nature(data_0)
    # 处理Nan
    data_0.bfill(inplace=True)  # 用后一个非NaN值填充（后向填充）
    data_0.ffill(inplace=True)  # 用前一个非NaN值填充（前向填充）

    # 20251005 增加2分类代码
    if classification == 2:
        if classification_direction == "buy":
            labeled = labeled[labeled['label'].isin([1, 2])]
        else:
            labeled = labeled[labeled['label'].isin([3, 4])]

    # 处理样本不均
    local_data_dict = {k: [] for k in temp_data_dict.keys()}
    local_label_list = []
    local_time_list = []

    # 遍历 concat_labeled 数据
    for labeled_time, labeled_row in labeled.iterrows():
        """1、寻找本级别匹配行"""
        data_0_filter = (data_0.index == labeled_time)
        matched_0 = data_0[data_0_filter]
        if len(matched_0) > 0:
            matched_0_index = matched_0.index[-1]
            matched_0_row_index = data_0.index.get_loc(matched_0_index)
            if matched_0_row_index >= time_point_step:
                # # 时间索引用完了，跟tea_radical_nature_label3一样删掉它，不然计算指标会报错
                # data_0_tmp = data_0.iloc[matched_0_row_index - time_point_step: matched_0_row_index].reset_index(
                #     drop=True)
                # 列表截断时含头不含尾，错过了信号当天的bar，下面修正一下
                start = matched_0_row_index - time_point_step + 1
                end = matched_0_row_index + 1
                if start < 0:
                    continue
                data_0_tmp = data_0.iloc[start:end].reset_index(drop=True)

                if feature_plan_name == 'feature_all':
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        print("还有Nan")
                        continue  # 数据NaN，跳过
                    local_data_dict['ema10'].append(ema10)
                    local_data_dict['ema20'].append(ema20)
                    local_data_dict['ema60'].append(ema60)
                    local_data_dict['macd'].append(macd)
                    local_data_dict['signal'].append(signal)
                    local_data_dict['adx'].append(adx)
                    local_data_dict['plus_di'].append(plus_di)
                    local_data_dict['minus_di'].append(minus_di)
                    local_data_dict['atr'].append(atr)
                    local_data_dict['boll_mid'].append(boll_mid)
                    local_data_dict['boll_upper'].append(boll_upper)
                    local_data_dict['boll_lower'].append(boll_lower)
                    local_data_dict['rsi'].append(rsi)
                    local_data_dict['obv'].append(obv)
                    local_data_dict['volume_ma5'].append(volume_ma5)
                    local_data_dict['volume'].append(volume)
                    local_data_dict['close'].append(close)
                elif feature_plan_name == 'feature_extremum':
                    open = data_0_tmp["open"]
                    high = data_0_tmp["high"]
                    low = data_0_tmp["low"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (open.isna().any() or high.isna().any() or low.isna().any() or close.isna().any()
                            or volume.isna().any()):
                        continue  # 数据NaN，跳过
                    local_data_dict['volume'].append(volume)
                    local_data_dict['open'].append(open)
                    local_data_dict['high'].append(high)
                    local_data_dict['low'].append(low)
                    local_data_dict['close'].append(close)
                elif feature_plan_name == 'feature_tea_radical_nature':
                    ret_5 = data_0_tmp["ret_5"]
                    hl_range = data_0_tmp["hl_range"]
                    upper_wick_pct = data_0_tmp["upper_wick_pct"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    dist_to_low_20 = data_0_tmp["dist_to_low_20"]
                    dist_to_high_20 = data_0_tmp["dist_to_high_20"]
                    range_pos_20 = data_0_tmp["range_pos_20"]

                    check_cols = [
                        ret_5, hl_range, upper_wick_pct, range_pos_20, dist_to_low_20, dist_to_high_20, volume, close
                    ]
                    if any(s.isna().any() for s in check_cols):
                        continue

                    cols = [
                        "ret_5",
                        "hl_range",
                        "upper_wick_pct",
                        'dist_to_low_20',
                        'dist_to_high_20',
                        'range_pos_20',
                        "volume",
                        "close"
                    ]
                    arr = data_0_tmp[cols].to_numpy(dtype=float)
                    if not np.isfinite(arr).all():
                        continue

                    local_data_dict['ret_5'].append(ret_5)
                    local_data_dict['hl_range'].append(hl_range)
                    local_data_dict['upper_wick_pct'].append(upper_wick_pct)
                    local_data_dict['dist_to_low_20'].append(dist_to_low_20)
                    local_data_dict['dist_to_high_20'].append(dist_to_high_20)
                    local_data_dict['range_pos_20'].append(range_pos_20)
                    local_data_dict['volume'].append(volume)
                    local_data_dict['close'].append(close)

                elif feature_plan_name == 'feature_basic_plus':
                    ret_5 = data_0_tmp["ret_5"]
                    hl_range = data_0_tmp["hl_range"]
                    upper_wick_pct = data_0_tmp["upper_wick_pct"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]

                    dist_to_low_20 = data_0_tmp["dist_to_low_20"]
                    dist_to_high_20 = data_0_tmp["dist_to_high_20"]
                    range_pos_20 = data_0_tmp["range_pos_20"]

                    check_cols = [
                        ret_5, hl_range, upper_wick_pct, range_pos_20, dist_to_low_20, dist_to_high_20, volume, close
                    ]

                    if any(s.isna().any() for s in check_cols):
                        continue
                    cols = [
                        "ret_5",
                        "hl_range",
                        "upper_wick_pct",
                        'dist_to_low_20',
                        'dist_to_high_20',
                        'range_pos_20',
                        "volume",
                        "close"
                    ]
                    arr = data_0_tmp[cols].to_numpy(dtype=float)
                    if not np.isfinite(arr).all():
                        continue
                    local_data_dict['ret_5'].append(ret_5)
                    local_data_dict['hl_range'].append(hl_range)
                    local_data_dict['upper_wick_pct'].append(upper_wick_pct)
                    local_data_dict['dist_to_low_20'].append(dist_to_low_20)
                    local_data_dict['dist_to_high_20'].append(dist_to_high_20)
                    local_data_dict['range_pos_20'].append(range_pos_20)
                    local_data_dict['volume'].append(volume)
                    local_data_dict['close'].append(close)
                elif feature_plan_name in FEATURE_PLAN_SPECS:
                    cols = FEATURE_PLAN_SPECS[feature_plan_name]
                    arr = data_0_tmp[cols].to_numpy(dtype=float)
                    if not np.isfinite(arr).all():
                        continue

                    for col in cols:
                        local_data_dict[col].append(data_0_tmp[col])
            else:
                continue  # backtest_bar 越界，跳过
        else:
            continue  # 无匹配日期，跳过
        local_label_list.append(int(labeled_row['label']))
        local_time_list.append(labeled_time)

    # # 先看有效样本的真实类别分布
    # print("有效样本数:", len(local_label_list))
    # print(pd.Series(local_label_list).value_counts())

    if handle_uneven_samples:
        if classification == 2:
            if classification_direction == "buy":
                expected_labels = [1, 2]
            else:
                expected_labels = [3, 4]

            keep_idx = _balance_indices_by_label_generic(
                local_label_list,
                local_time_list,
                expected_labels=expected_labels,
                require_all=True
            )

        elif classification == 4:
            expected_labels = [1, 2, 3, 4]
            keep_idx = _balance_indices_by_label_generic(
                local_label_list,
                local_time_list,
                expected_labels=expected_labels,
                require_all=True
            )
        else:
            keep_idx = list(range(len(local_label_list)))

        if keep_idx is None or len(keep_idx) == 0:
            # print(assetList[0].assetsCode, "有效样本均衡后无法继续，终止")
            return None
    else:
        keep_idx = list(range(len(local_label_list)))

    # # 查看样本均衡后的分布
    # if keep_idx is not None and len(keep_idx) > 0:
    #     balanced_labels = [local_label_list[i] for i in keep_idx]
    #     print(assetList[0].assetsCode, "均衡后标签分布：")
    #     print(pd.Series(balanced_labels).value_counts().sort_index())

    for i in keep_idx:
        for k in local_data_dict.keys():
            temp_data_dict[k].append(local_data_dict[k][i])
        temp_label_list.append(local_label_list[i])

    # print(assetList[0].assetsCode, "结束", len(temp_label_list))
    # print(labeled['label'].value_counts())
    # print(pd.Series(temp_label_list).value_counts())

    return "success"


def up_time_level_point_to_ts(assetList, temp_data_dict, temp_label_list, time_point_step, handle_uneven_samples,
                              strategy_name, label_name, feature_plan_name, up_time_level):
    # 加载数据
    if strategy_name == 'identify_Market_Types':
        item = 'market_condition_backtest'
    else:
        item = 'trade_point_backtest_' + strategy_name
    labeled_filePath = (SQTools.read_config("SQData", item)
                        + assetList[0].assetsMarket
                        + "_"
                        + assetList[0].assetsCode
                        + "_"
                        + assetList[0].barEntity.timeLevel
                        + str(label_name)
                        + ".csv")
    if up_time_level == '15':
        data_up_filePath = (SQTools.read_config("SQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + assetList[0].assetsCode
                            + "_"
                            + '15'
                            + ".csv")
    elif up_time_level == '30':
        data_up_filePath = (SQTools.read_config("SQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + assetList[0].assetsCode
                            + "_"
                            + '30'
                            + ".csv")
    elif up_time_level == '60':
        data_up_filePath = (SQTools.read_config("SQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + assetList[0].assetsCode
                            + "_"
                            + '60'
                            + ".csv")
    elif up_time_level == 'd':
        data_up_filePath = (SQTools.read_config("SQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + assetList[0].assetsCode
                            + "_"
                            + 'd'
                            + ".csv")
    elif up_time_level == 'index_d':
        data_up_filePath = (SQTools.read_config("SQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + "000001_index_d" + ".csv")
    else:
        raise ValueError("up_time_level输入异常")

    if not os.path.exists(labeled_filePath):
        return None
    labeled = pd.read_csv(labeled_filePath, index_col="time", parse_dates=True)
    if up_time_level == '15' or up_time_level == '30':
        data_0 = pd.read_csv(data_up_filePath, parse_dates=["time"])
    else:
        data_0 = pd.read_csv(data_up_filePath, index_col="time", parse_dates=True)

    # 计算所有指标
    data_0 = IMTHelper.calculate_indicators(data_0)
    # 处理Nan
    data_0.fillna(method='bfill', inplace=True)  # 用后一个非NaN值填充（后向填充）
    data_0.fillna(method='ffill', inplace=True)  # 用前一个非NaN值填充（前向填充）

    # 是否处理样本不均
    if handle_uneven_samples:
        handled_uneven_filepath = (SQTools.read_config("SQData", item)
                                   + assetList[0].assetsMarket
                                   + "_"
                                   + assetList[0].assetsCode
                                   + "_"
                                   + assetList[0].barEntity.timeLevel
                                   + str(label_name)
                                   + "_handled_uneven" + ".csv")
        if not os.path.exists(handled_uneven_filepath):
            if strategy_name == 'c4_oscillation_kdj_nature' or strategy_name == 'fuzzy_nature' \
                    or strategy_name == 'extremum':
                labeled = fuzzy_nature_handling_uneven_samples1(labeled)
            else:
                labeled = tea_radical_nature_handling_uneven_samples1(labeled)
            if labeled.empty:
                print(assetList[0].assetsCode, "处理样本不均终止")
                return None
            labeled.to_csv(handled_uneven_filepath, index=True)
        else:
            labeled = pd.read_csv(handled_uneven_filepath, index_col="time", parse_dates=True)

    # 遍历 concat_labeled 数据
    for labeled_time, labeled_row in labeled.iterrows():
        labeled_date = labeled_time.date()
        labeled_hour = labeled_time.hour

        if up_time_level == '15':
            data_0_start_time = labeled_time.floor("15T")  # 向下取整到最近的15分钟
            data_0_end_time = data_0_start_time + pd.Timedelta(minutes=15)
            # 在 15分钟级别 中查找符合区间的行
            data_0_mask = (data_0["time"] > data_0_start_time) & (data_0["time"] <= data_0_end_time)
            data_0_matching_indices = data_0.index[data_0_mask].to_numpy()  # 使用 to_numpy() 解决 FutureWarning

            if data_0_matching_indices.size > 0:
                data_0_prev_index = data_0_matching_indices[-1] - 1  # 获取上一行的索引

                # 检查是否越界
                if data_0_prev_index < 0:
                    continue

                # 获取上一行的time
                # data_0_prev_row_time = data_0.loc[data_0_prev_index, "time"]
                # data_0_row_index = data_0.index.get_loc(pd.Timestamp(data_0_prev_row_time))
                data_0_row_index = data_0_prev_index

                if data_0_row_index >= time_point_step:
                    data_0_tmp = data_0.iloc[data_0_row_index - time_point_step: data_0_row_index].reset_index(
                        drop=True)
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    continue  # d.csv 越界，跳过
            else:
                continue  # 无匹配日期，跳过
        elif up_time_level == '30':
            data_0_start_time = labeled_time.floor("30T")  # 向下取整到最近的15分钟
            data_0_end_time = data_0_start_time + pd.Timedelta(minutes=30)
            # 在 30分钟级别 中查找符合区间的行
            data_0_mask = (data_0["time"] > data_0_start_time) & (data_0["time"] <= data_0_end_time)
            data_0_matching_indices = data_0.index[data_0_mask].to_numpy()  # 使用 to_numpy() 解决 FutureWarning

            if data_0_matching_indices.size > 0:
                data_0_prev_index = data_0_matching_indices[-1] - 1  # 获取上一行的索引

                # 检查是否越界
                if data_0_prev_index < 0:
                    continue

                # 获取上一行的time
                # data_0_prev_row_time = data_0.loc[data_0_prev_index, "time"]
                # data_0_row_index = data_0.index.get_loc(pd.Timestamp(data_0_prev_row_time))
                data_0_row_index = data_0_prev_index

                if data_0_row_index >= time_point_step:
                    data_0_tmp = data_0.iloc[data_0_row_index - time_point_step: data_0_row_index].reset_index(
                        drop=True)
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    continue  # 越界，跳过
            else:
                continue  # 无匹配日期或小时，跳过
        elif up_time_level == '60':
            """60级别匹配行"""
            if labeled_hour == 9 or labeled_hour == 13:
                labeled_hour += 1  # 这俩匹配不上，只能改一下时间
            day_hour_filter = (data_0.index.date == labeled_date) & (data_0.index.hour == labeled_hour)
            matched_60 = data_0[day_hour_filter]
            if len(matched_60) > 0:
                matched_60_index = matched_60.index[-1]
                data_0_row_index = data_0.index.get_loc(matched_60_index)
                if data_0_row_index >= time_point_step:
                    data_0_tmp = data_0.iloc[data_0_row_index - time_point_step: data_0_row_index].reset_index(
                        drop=True)
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    continue  # backtest_bar 越界，跳过
            else:
                continue  # 无匹配日期，跳过
        elif up_time_level == 'd':
            if pd.Timestamp(labeled_date) in data_0.index:
                data_0_row_index = data_0.index.get_loc(pd.Timestamp(labeled_date))
                if data_0_row_index >= time_point_step:
                    data_0_tmp = data_0.iloc[data_0_row_index - time_point_step: data_0_row_index].reset_index(
                        drop=True)
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        print("还有Nan", data_0_row_index)
                        continue  # 数据NaN，跳过
                else:
                    print("越界", data_0_row_index)
                    continue  # backtest_bar 越界，跳过
            else:
                print("无匹配日期")
                continue  # 无匹配日期，跳过
        elif up_time_level == 'index_d':
            if pd.Timestamp(labeled_date) in data_0.index:
                data_0_row_index = data_0.index.get_loc(pd.Timestamp(labeled_date))
                if data_0_row_index >= time_point_step:
                    data_0_tmp = data_0.iloc[data_0_row_index - time_point_step: data_0_row_index].reset_index(
                        drop=True)
                    ema10 = data_0_tmp["ema10"]
                    ema20 = data_0_tmp["ema20"]
                    ema60 = data_0_tmp["ema60"]
                    macd = data_0_tmp["macd"]
                    signal = data_0_tmp["signal"]
                    adx = data_0_tmp["adx"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    atr = data_0_tmp["atr"]
                    boll_mid = data_0_tmp["boll_mid"]
                    boll_upper = data_0_tmp["boll_upper"]
                    boll_lower = data_0_tmp["boll_lower"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume_ma5 = data_0_tmp["volume_ma5"]
                    close = data_0_tmp["close"]
                    volume = data_0_tmp["volume"]
                    if (ema10.isna().any() or ema20.isna().any() or ema60.isna().any()
                            or macd.isna().any() or signal.isna().any() or adx.isna().any()
                            or plus_di.isna().any() or minus_di.isna().any() or atr.isna().any()
                            or boll_mid.isna().any() or boll_upper.isna().any() or boll_lower.isna().any()
                            or rsi.isna().any() or obv.isna().any() or volume_ma5.isna().any()
                            or close.isna().any() or volume.isna().any()):
                        continue  # 数据NaN，跳过
                else:
                    continue  # backtest_bar 越界，跳过
            else:
                continue  # 无匹配日期，跳过

        # # 如果通过所有越界检查，将数据存入字典  标签存入列表
        temp_data_dict['ema10'].append(ema10)
        temp_data_dict['ema20'].append(ema20)
        temp_data_dict['ema60'].append(ema60)
        temp_data_dict['macd'].append(macd)
        temp_data_dict['signal'].append(signal)
        temp_data_dict['adx'].append(adx)
        temp_data_dict['plus_di'].append(plus_di)
        temp_data_dict['minus_di'].append(minus_di)
        temp_data_dict['atr'].append(atr)
        temp_data_dict['boll_mid'].append(boll_mid)
        temp_data_dict['boll_upper'].append(boll_upper)
        temp_data_dict['boll_lower'].append(boll_lower)
        temp_data_dict['rsi'].append(rsi)
        temp_data_dict['obv'].append(obv)
        temp_data_dict['volume_ma5'].append(volume_ma5)
        temp_data_dict['volume'].append(volume)
        temp_data_dict['close'].append(close)

        temp_label_list.append(labeled_row['label'])

    print(assetList[0].assetsCode, "结束", len(temp_label_list))
    return "success"


def get_feature(feature_plan_name):
    # 创建一个字典来存储匹配的结果
    if feature_plan_name == 'feature_all':
        return {'ema10': [], 'ema20': [], 'ema60': [], 'macd': [], 'signal': [],
                'adx': [], 'plus_di': [], 'minus_di': [], 'atr': [], 'boll_mid': [], 'boll_upper': [],
                'boll_lower': [], 'rsi': [], 'obv': [], 'volume_ma5': [], 'close': [], 'volume': []
                }
    elif feature_plan_name == 'feature_tea_radical_nature':
        return {
            'ret_5': [],
            'hl_range': [],
            'upper_wick_pct': [],
            'dist_to_low_20': [],
            'dist_to_high_20': [],
            'range_pos_20': [],
            'volume': [],
            'close': []
        }
    elif feature_plan_name == 'feature_basic_plus':
        return {
            'ret_5': [],
            'hl_range': [],
            'upper_wick_pct': [],

            'dist_to_low_20': [],
            'dist_to_high_20': [],
            'range_pos_20': [],

            'volume': [],
            'close': []
        }
    elif feature_plan_name == 'feature_extremum':
        return {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    elif feature_plan_name == 'feature_tea_concat':
        return {'index_d_close': [], 'index_d_volume': [], 'd_close': [], 'd_volume': [], 'close_60': [],
                'volume_60': []}
    elif feature_plan_name == 'feature_tea_multi_level':
        return {'MACD_0': [], 'DIF_0': [], 'DEA_0': [], 'K_0': [], 'D_0': [], 'J_0': [], 'close_0': [],
                'MACD_1': [], 'DIF_1': [], 'DEA_1': [], 'K_1': [], 'D_1': [], 'J_1': [], 'close_1': [],
                'MACD_2': [], 'DIF_2': [], 'DEA_2': [], 'K_2': [], 'D_2': [], 'J_2': [], 'close_2': []
                }
    elif feature_plan_name == 'feature_fuzzy_multi_level':
        return {'MACD_0': [], 'DIF_0': [], 'MA_5_0': [], 'MA_10_0': [], 'MA_60_0': [], 'close_0': [],
                'MACD_1': [], 'DIF_1': [], 'MA_5_1': [], 'MA_10_1': [], 'MA_60_1': [], 'close_1': [],
                'MACD_2': [], 'DIF_2': [], 'MA_5_2': [], 'MA_10_2': [], 'MA_60_2': [], 'close_2': []
                }
    elif feature_plan_name in FEATURE_PLAN_SPECS:
        return {col: [] for col in FEATURE_PLAN_SPECS[feature_plan_name]}
    else:
        print("未指定feature_plan_name")
        return None


def get_point_to_ts(time_point_step, handle_uneven_samples, strategy_name,
                    feature_plan_name, p2t_name, label_name, temp_data_dict, temp_label_list, assetList, up_time_level,
                    classification, classification_direction):
    res = None
    if (strategy_name == 'c4_trend_nature'
            or strategy_name == 'c4_oscillation_boll_nature'
            or strategy_name == 'c4_oscillation_kdj_nature'
            or strategy_name == 'c4_breakout_nature'
            or strategy_name == 'c4_reversal_nature'
            or strategy_name == 'tea_radical_nature'
            or strategy_name == 'fuzzy_nature'
            or strategy_name == 'fuzzy_ma'
            or strategy_name == 'identify_Market_Types'
            or strategy_name == 'extremum'):
        if p2t_name == "point_to_ts_single":  # 单级别
            res = single_time_level_point_to_ts(assetList, temp_data_dict, temp_label_list, time_point_step,
                                                handle_uneven_samples,
                                                strategy_name, label_name, feature_plan_name,
                                                classification, classification_direction)
        elif p2t_name == "point_to_ts_up_time_level":  # 找上级行情
            res = up_time_level_point_to_ts(assetList, temp_data_dict, temp_label_list, time_point_step,
                                            handle_uneven_samples,
                                            strategy_name, label_name, feature_plan_name, up_time_level)
        elif p2t_name == "point_to_ts_concat":
            # 拼接所有点，60、d、index_d造数
            res = tea_radical_nature_point_to_ts3(assetList, temp_data_dict, temp_label_list, time_point_step,
                                                  handle_uneven_samples,
                                                  strategy_name, label_name, feature_plan_name)
        elif p2t_name == "point_to_ts_tea_multi_level":
            # 5标注，5、15、30造数
            res = tea_radical_nature_point_to_ts2(assetList, temp_data_dict, temp_label_list, time_point_step,
                                                  handle_uneven_samples,
                                                  strategy_name, label_name, feature_plan_name)
        elif p2t_name == "point_to_ts_fuzzy_multi_level":
            # 30标注，30、60、d造数
            res = fuzzy_nature_point_to_ts2(assetList, temp_data_dict, temp_label_list, time_point_step,
                                            handle_uneven_samples,
                                            strategy_name, label_name, feature_plan_name)
    else:
        raise ValueError("strategy_name输入异常")

    return res, temp_data_dict, temp_label_list


def prepare_dataset(flag, name, time_point_step, limit_length, handle_uneven_samples, strategy_name,
                    feature_plan_name, plan_count, p2t_name, label_name, classification, classification_direction):
    """
    A股 a800_stocks
    row['code'][3:]  row['code']  '5', '15', '30', '60', 'd'  stock  1  A

    港股 hk_1000_stock_codes
    row['code']  row['code']  'd'  stock  1  HK

    美股 sp500_stock_codes
    row['code']  row['code']  'd'  stock  1  USA

    数字币 crypto_code  crypto_code_wait_handle_stocks
    row['code']  row['code']  '15', '60', '240', 'd'  crypto  1  crypto
    """
    asset_code_path = SQTools.read_config("SQT", "asset_code")
    allStockCode = pd.read_csv(asset_code_path + "a800_stocks_2025.csv", dtype={'code': str})
    allStockCode_shuffled = allStockCode.sample(frac=1, random_state=42).reset_index(drop=True)

    if flag == "_TRAIN":
        df_dataset = allStockCode_shuffled.iloc[:500]
    else:
        df_dataset = allStockCode_shuffled.iloc[500:]

    temp_data_dict = get_feature(feature_plan_name)
    temp_label_list = []
    """
    插播一下，ts文件写入卡了我2天
    Time-Series-Library库里有除了时序预测也有时序分类，github主页给了时序分类的数据集地址，我下载到了D:\github\dataset\classification
    打开发现数据是.ts文件，找到人家官网https://www.timeseriesclassification.com/，发现个aeon的库，aeon是个专门处理时序数据的库，包括
    组织数据，调算法，可看作scikit-learn加强版。然后我学习了aeon组织数据的方法，知道了一个股票应该
    组成(400000, 2, 500)，40万行，每行close、volume2个特征，每个特征500时间步。下面折叠了1，有兴趣可以打开看。。。
    我不知道集成学习代码怎么写，于是想用Time-Series-Library封装好的，于是决定日线、小时线、大盘指数 数据集三合一，组成(400000, 6, 500)
    好开始组装数据，我发现Time-Series-Library读取ts文件用的是UEALoader工具，但这工具不是出自aeon库，而是sktime库，得，aeon怎么组织数据白看了
    我先看ts文件怎么组织数据的，下面折叠了2，有兴趣可以打开看。。。
    又断点看了UEALoader读取ts文件逻辑,下面折叠了3，有兴趣可以打开看。。。
    看完知道怎么读了，不知道怎么写入ts文件，网上找不到，chatgpt胡言乱语，最后无意发现sktime库有个write_dataframe_to_tsfile函数
    用chatgpt试了各种报错，想放弃，写了个写入csv的trans_labeled_point_to_ts_bak
    又断点看代码，终于调通，核心就是 一行数据，6个特征，每个特征是500时间步，时间步是截取的df，这时要删除原来的时间索引，就变成ndarray格式，
    append到字典里，不要label列，label单独append到list里。所有数据append完，字典转df，list转Series，给入参函数
    ts文件生成后，我模仿人家的数据集，在ts文件上面加了个注释@dimensions 6 代表特征数。函数不知道是不是更新了没有这个入参

    另外，ts文件中，序列是否等长@equalLength false，这个序列我也不知道是什么，反正他们ts文件中，一行数据每个特征的序列的时间步是一样长的，
    但行与行之间的序列的时间步不一样，我都是500，不涉及这个问题，但这位交易对提供了可能性，比如这行6个特征都抽500步，下一行抽270步。
    但是，一行数据日线抽500，大盘抽250是不行的，折叠了3里说了原因
    aeon里讲过变长序列问题，下面折叠了4，有兴趣可以打开看。。。但Time-Series-Library没用aeon，所以看也没用，
    但是，Time-Series-Library在exp.train时读取batch_x，是(16,29,12)，批量大小*本批次时间步序列最大值*12个特征，折叠3读ts文件里说过读取
    的日语train的ts文件，270行，12个特征，第一行每个特征的series对象长20，第二行的长25。。。  每行的series竖着展开，加权270行，就是4274，
    12列，每列4274，整体就是4274*12，  批量在取数据时，270里取了随机16行，一行是 (1,20多,12)，16行就是(16,20多,12)，16行找时间步最长的，就是
    (16,29,12)
    对我来说，40多万行随机取16行，一行是(1,500,6),16行就是(16,500,6)，进入我的cnn，

    为了方便调试，我把时间步从500改为5，但我的数据用其他模型跑报错，断点对比了很久，发现是时间步最少是8，改成10不报错了
    但我的模型应该接收(16, 6, 1, 500)这种格式，batch_x是三维的，我调整为4维，不报错了
    ------------------------------
    1
        数据组装格式  
        https://www.aeon-toolkit.org/en/stable/examples/datasets/datasets.html
        他们的方式是(n_cases, n_channels, n_timepoints)  样本数，特征数，时间点
        在一个时间点观察到一个值，比如500天的日线收盘价就是 （1，500），用X表示  标记为有效买，那y就是1，若有连续5个交易点，那么
            X = np.random.random((5, 1, 500))
            y = np.array([1, 2, 1, 3, 4])  对应我四个分类：1有效买入，2无效买入，3有效卖出，4无效卖出
        在一个时间点观察到一个向量，比如500天的日线 收盘价+成交量，那就是
            X = np.random.random((5, 2, 500))  y不变
        一个股票我有500多个点位，800个股票有40多万个点位，我的
            日线 X_day = np.random.random((400000, 2, 500))
            小时线 X_60 = np.random.random((400000, 2, 500))
            大盘日线 X_day_sz000001_index = np.random.random((400000, 2, 500))
    ------------------------------
    2
        UEA，时间序列分类，不含时间戳，一行数据是多个特征，用冒号分开，最后一个冒号后面是分类。
        日线 X_day = np.random.random((400000, 2, 500))
        这在UEA里，就算40万行，每行前面是500个close用逗号隔开，然后冒号，后面500个volume用逗号隔开，最后冒号，最后分类，一行是1000多个值
            我这1000一行不多，他还有一行数据10万，900多个特征
    3
        UEALoader读取ts文件逻辑：
        以JapaneseVowels为例，12个特征，变长，意味着每行12个冒号，2个冒号之间有多少值不固定。
        load_from_tsfile_to_dataframe先读出df、labels
        df是三维，12个特征 * 270 * 每个冒号的变长序列 
        label 是270*1 但各个类别的数据都放在一起了

        然后df 三维  转化为二维 ，原来 是 270 * 12  每一行这12个特征序列长度相同，现在把序列变为列，相当于第一行变长20行，第二行变长26行，等等
        所以对于每一列来说，都是有4272行数据， 这4272不能除以270，而是有270行变长序列展开后加起来的  
    4    
        关于变长序列处理办法：
        您可以将序列填充到最长的长度，或者如果长度不相等，则可以将它们截断为集合中最短的长度序列。
        对于分类问题，数据用序列均值填充，并添加了低级高斯噪声。
        加载等长是默认行为
        https://www.aeon-toolkit.org/en/stable/examples/datasets/data_unequal.html

        我本来想用过滤出交易对，优点：一买对应一卖，统计收益率方便。缺点：有效点位减少2/3可能干扰模型（某个时间段都是有效买入区间）、变长序列处理方式可能干扰模型。
        目前存在连续多个买入，连续多个卖出，可通过仓位管理控制，不再苛求策略。缺点：收益率统计要再想办法
    """
    for index, row in df_dataset.iterrows():
        assetList = SQAsset.asset_generator(row['code'][3:],
                                            row['code'],
                                            ['d'],
                                            'stock',
                                            1, 'A')
        # 准备训练数据
        res, temp_data_dict, temp_label_list = get_point_to_ts(time_point_step, handle_uneven_samples, strategy_name,
                                                               feature_plan_name, p2t_name, label_name, temp_data_dict,
                                                               temp_label_list, assetList, None,
                                                               classification, classification_direction)
        if not res:
            continue
        if limit_length == 0:  # 全数据
            pass
        elif len(temp_label_list) >= limit_length:  # 只要部分数据
            break
    # 循环结束后，字典转为DataFrame
    lens = {k: len(v) for k, v in temp_data_dict.items()}
    print(feature_plan_name, "各列长度：", lens)

    if len(set(lens.values())) != 1:
        print("长度不一致的列：")
        for k, v in lens.items():
            print(k, v)
        raise ValueError("temp_data_dict 各列长度不一致，无法生成 DataFrame")

    result_df = pd.DataFrame(temp_data_dict)
    # 将列表转换成 Series
    result_series = pd.Series(temp_label_list)
    """
    # 创建一个符合要求的 DataFrame
    data = {
        "feature1": [pd.Series([1, 2, 3, 4]), pd.Series([5, 6, 7, 8])],
        "feature2": [pd.Series([4, 5, 6, 7]), pd.Series([1, 2, 3, 4])]
    }
    """
    if plan_count is not None:
        temp_file_feature_name = plan_count
    else:
        temp_file_feature_name = feature_plan_name

    if classification == 4:
        problem_name_str = ("_" + name + "_" + str(strategy_name) + "_" + str(temp_file_feature_name) + "_" +
                            str(handle_uneven_samples) + "U" + str(label_name) + "_" + str(time_point_step) + "S")
        if strategy_name == "identify_Market_Types":
            class_value_list_str = ["1", "2", "3"]
        else:
            class_value_list_str = ["1", "2", "3", "4"]
    elif classification == 2:
        problem_name_str = ("_" + name + "_" + str(strategy_name) + "_" + str(temp_file_feature_name) + "_" +
                            str(handle_uneven_samples) + "U" + str(label_name) + "_" + str(time_point_step) +
                            "S_" + str(classification) + "C_" + str(classification_direction))
        if classification_direction == 'buy':
            class_value_list_str = ["1", "2"]
        else:
            class_value_list_str = ["3", "4"]
    # 写入 ts 文件
    write_dataframe_to_tsfile(
        data=result_df,
        path="D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts",  # 保存文件的路径
        problem_name=problem_name_str,  # 问题名称
        class_label=class_value_list_str,  # 是否有 class_label
        class_value_list=result_series,  # 是否有 class_label
        equal_length=True,
        fold=flag
    )


def prepare_dataset_single(flag, name, time_point_step, limit_length, handle_uneven_samples, strategy_name,
                           feature_plan_name, p2t_name, label_name, count, pred_market_type, up_time_level):
    asset_code_path = SQTools.read_config("SQT", "asset_code")
    allStockCode = pd.read_csv(asset_code_path + "a800_stocks.csv", dtype={'code': str})
    df_dataset = allStockCode.iloc[500:]
    n = 1
    for index, row in df_dataset.iterrows():
        temp_data_dict = get_feature(feature_plan_name)
        temp_label_list = []
        assetList = SQAsset.asset_generator(row['code'][3:],
                                            row['code_name'],
                                            ['15'],  # 找上级也填单级别，因为上级在up_time_level_point_to_ts里写死了
                                            'stock',
                                            1, 'A')
        # 准备训练数据
        res, temp_data_dict, temp_label_list = get_point_to_ts(time_point_step, handle_uneven_samples, strategy_name,
                                                               feature_plan_name, p2t_name, label_name, temp_data_dict,
                                                               temp_label_list, assetList, up_time_level)
        # 循环结束后，字典转为DataFrame
        result_df = pd.DataFrame(temp_data_dict)
        # 将列表转换成 Series
        result_series = pd.Series(temp_label_list)

        problem_name_str = ("pred_" + str(pred_market_type) + "_" + name + "_"
                            + assetList[0].assetsMarket + "_"
                            + assetList[0].assetsCode + "_" + assetList[0].barEntity.timeLevel
                            + "_" + str(strategy_name) + "_" + str(feature_plan_name) + "_"
                            + str(handle_uneven_samples) + "_uneven" + str(label_name) + "_"
                            + str(time_point_step) + "step")
        if strategy_name == "identify_Market_Types":
            class_value_list_str = ["1", "2", "3"]
        else:
            class_value_list_str = ["1", "2", "3", "4"]

        # 20250228增加逻辑：如果是为策略交易点判断当时的行情类型，则改为行情分类
        # 注意，不能在get_point_to_ts中挑出1、3类，因为预测结果是list，没有时间，跟原始label文件对不上
        # if pred_market_type:
        #     class_value_list_str = ["1", "2", "3"]
        #
        #     if len(temp_label_list) <= 3:
        #         # 修改前三个值为 "1", "2", "3"，其余值改为 "3"
        #         print(assetList[0].assetsCode, "数据量少于3条，不值得测试")
        #         continue
        #     else:
        #         temp_label_list = ["1", "2", "3"] + ["3"] * (len(temp_label_list) - 3)
        #         # 预测行情，不计算准确率，所以原来的买卖分类无所谓，随便填
        #         result_series = pd.Series(temp_label_list)

        # 写入 ts 文件
        write_dataframe_to_tsfile(
            data=result_df,
            path="D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts/prediction",  # 保存文件的路径
            problem_name=problem_name_str,  # 问题名称
            class_label=class_value_list_str,  # 是否有 class_label
            class_value_list=result_series,  # 是否有 class_label
            equal_length=True,
            fold=flag
        )
        # 写入 ts 文件
        write_dataframe_to_tsfile(
            data=result_df,
            path="D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts/prediction",  # 保存文件的路径
            problem_name=problem_name_str,  # 问题名称
            class_label=class_value_list_str,  # 是否有 class_label
            class_value_list=result_series,  # 是否有 class_label
            equal_length=True,
            fold="_TRAIN"
        )
        # n += 1
        # if n > count:
        #     break


def concat_trade_point(assetList, strategy_name):
    # 读取交易点
    item = 'trade_point_backtest_' + strategy_name
    tpl_filepath = (SQTools.read_config("SQData", item)
                    + assetList[0].assetsMarket
                    + "_")

    df_tpl_5 = pd.read_csv(tpl_filepath +
                           assetList[0].assetsCode + "_" + assetList[0].barEntity.timeLevel + ".csv")
    df_tpl_15 = pd.read_csv(tpl_filepath +
                            assetList[1].assetsCode + "_" + assetList[1].barEntity.timeLevel + ".csv")
    df_tpl_30 = pd.read_csv(tpl_filepath +
                            assetList[2].assetsCode + "_" + assetList[2].barEntity.timeLevel + ".csv")
    df_tpl_60 = pd.read_csv(tpl_filepath +
                            assetList[3].assetsCode + "_" + assetList[3].barEntity.timeLevel + ".csv")

    df_tpl_d = None
    # temp2中有16个股票是单边行情，没用日线交易信号
    try:
        df_tpl_d = pd.read_csv(tpl_filepath +
                               assetList[4].assetsCode + "_" + assetList[4].barEntity.timeLevel + ".csv")
    except Exception as e:
        pass

    # 整合所有交易点
    if df_tpl_d is not None:
        df_tpl_d['time'] = pd.to_datetime(df_tpl_d['time'])  # 将 time 列转换为日期时间类型
        # 筛选出时间在 2019年1月1日及之后的行  因为分钟级别数据都是从这天开始，日线数据太久远影响收益率
        df_tpl_d_filtered = df_tpl_d[df_tpl_d['time'] >= '2019-01-01']
        df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60, df_tpl_d_filtered], ignore_index=True)
    else:
        df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60], ignore_index=True)

    # 将第一列转换为 datetime 格式
    df_tpl = df_tpl.set_index(df_tpl.columns[0])  # 使用第一列作为索引
    df_tpl.index.name = 'time'  # 将索引命名为 'time'
    # 修改时间列格式（索引）
    df_tpl.index = pd.to_datetime(df_tpl.index)
    # 修改时间列中含有00:00:00的部分为15:00:00
    df_tpl.index = df_tpl.index.map(
        lambda x: x.replace(hour=15, minute=0, second=0) if x.hour == 0 and x.minute == 0 and x.second == 0 else x)
    # 修改其余列的名称
    # df_tpl.columns = ['price', 'signal']
    # 按索引（时间）排序
    df_tpl = df_tpl.sort_index()
    # 保存为新的 CSV 文件
    df_tpl.to_csv(tpl_filepath + assetList[0].assetsCode + "_concat" + ".csv")


def prepare_train_dataset():
    """"""
    """
    标注完成，准备训练数据
    由于两个数据集都要做，因此写俩方法串行，别删
        按照原始标注方法，_TRAIN 最多24.6万  _TEST 最多14.6万
        limit_length==0 代表不截断，全数据

        flag: _TRAIN 训练集截前500个股票，  _TEST 测试集截后300个  截之前按固定随机数乱序了
        time_point_step: 截取的时间步，最长500，最少得是8以上，因为很多时序模型需要得序列长度最少是8
        limit_length：限制长度是为了方便debug时调试，数据太多加载太慢
        handle_uneven_samples: macd策略样本不均，其他策略不一定有这个问题，所以这里控制要不要处理
        strategy_name: 为了读回测点文件，     
                        identify_Market_Types  不需要处理样本不均
                        fuzzy_nature
                        tea_radical_nature
                        c4_trend_nature
                        c4_oscillation_boll_nature
                        c4_oscillation_kdj_nature
                        c4_breakout_nature
                        c4_reversal_nature
                        extremum  不需要处理样本不均
        feature_plan_name: 不同特征组织方案
                feature_tea_concat 日线、小时线、指数日线
                feature_tea_multi_level macd5分钟、15分钟、30分钟
                feature_fuzzy_multi_level
                feature_all
                feature_tea_macd
                feature_c4_trend
                feature_c4_oscillation_boll
                feature_c4_oscillation_kdj
                feature_c4_breakout
                feature_c4_reversal  
                feature_extremum
        p2t_name: point_to_ts_single
                  point_to_ts_up_time_level
                  point_to_ts_concat  拼接所有点，60、d、index_d造数
                  point_to_ts_tea_multi_level  5标注，5、15、30造数
                  point_to_ts_fuzzy_multi_level  30标注，30、60、d造数
        label_name: 合并标注交易点的  此时flag只会是 _concat_label1
                    各级别标注交易点  "_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                    fuzzy的各级别flag也有 _label1
        name: 标的_级别 ts文件命名，跟limit_length对应，这文件有多少条数据
                跑单级别时，在Dataset里只填对应级别        
    """
    prepare_dataset("_TRAIN", "A_d", 160,
                    50000, True,
                    "tea_radical_nature", "feature_all",
                    None, "point_to_ts_single", "_label5", 2, "buy")
    prepare_dataset("_TEST", "A_d", 160,
                    10000, True,
                    "tea_radical_nature", "feature_all",
                    None, "point_to_ts_single", "_label5", 2, "buy")


def prepare_pred_dataset():
    """""" """
    组装预测数据  
        预测交易点:
            point_to_ts_single  用本级别交易点，找对应时间回测数据，策略和特征可随意组合
            pred_market_type False
            up_time_level 任意值都行
        预测行情
            point_to_ts_up_time_level  用本级别交易点，找up_time_level对应级别的回测数据，
            特征只用feature_all

    20251003 处理c4_oscillation_boll_nature 和 c4_oscillation_kdj_nature时报文件不存在，原因是文件夹名字太长了，在Dataset.py
    里把problem_name_str里的策略名改短即可
    
    20260603 若需要组装的特征，输入名字即可，比如ftr_2_dist_to_low_20_range_pos_20
    """
    print(FEATURE_PLAN_SPECS)

    prepare_dataset_single("_TEST", "A_15", 160,
                           20000, True,
                           "fuzzy_nature", "feature_all",
                           "point_to_ts_single", "_label1", None,
                           False, 'd')
