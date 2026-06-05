import os
import pandas as pd
import numpy as np
from sktime.datasets import write_dataframe_to_tsfile
from itertools import combinations

import SQData.Asset as SQAsset
import SQTool.Tools as SQTools
import SQData.Identify_market_types_helper as IMTHelper

# 你的基础通道：每个计划都保留
BASE_FEATURES = [
    'ema10', 'ema20', 'ema60',
    'macd', 'signal',
    'adx', 'plus_di', 'minus_di',
    'atr',
    'boll_mid', 'boll_upper', 'boll_lower',
    'rsi', 'obv', 'volume_ma5',
    'close', 'volume'
]

# 自己攒的特征
USUAL_FEATURES = [
    'ret_5', 'hl_range', 'upper_wick_pct'
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


# 只注册你想扫的这一类
register_combo_plans(
    base_name="ftr",
    base_cols=BASE_FEATURES,
    pool=INTERACTION_CANDIDATES,
    combo_sizes=(1, 2)
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
                # 列表截断时含头不含尾，错过了信号当天的bar，下面修正一下
                start = matched_0_row_index - time_point_step + 1
                end = matched_0_row_index + 1
                if start < 0:
                    continue
                data_0_tmp = data_0.iloc[start:end].reset_index(drop=True)

                if feature_plan_name == 'feature_tea_radical_nature':
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
                elif feature_plan_name == 'feature_fuzzy_ma':
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]
                    volume = data_0_tmp["volume"]
                    close = data_0_tmp["close"]
                    close_ma5_ratio = data_0_tmp["close_ma5_ratio"]
                    range_pos_20 = data_0_tmp["range_pos_20"]
                    body_to_range = data_0_tmp["body_to_range"]
                    check_cols = [
                        plus_di, minus_di, rsi, obv, close_ma5_ratio, range_pos_20, body_to_range,
                        volume, close
                    ]
                    if any(s.isna().any() for s in check_cols):
                        continue
                    cols = [
                        'plus_di',
                        'minus_di',
                        'rsi',
                        'obv',
                        "volume",
                        "close",
                        "close_ma5_ratio",
                        "range_pos_20",
                        "body_to_range"
                    ]
                    arr = data_0_tmp[cols].to_numpy(dtype=float)
                    if not np.isfinite(arr).all():
                        continue
                    local_data_dict['plus_di'].append(plus_di)
                    local_data_dict['minus_di'].append(minus_di)
                    local_data_dict['rsi'].append(rsi)
                    local_data_dict['obv'].append(obv)
                    local_data_dict['volume'].append(volume)
                    local_data_dict['close'].append(close)
                    local_data_dict['close_ma5_ratio'].append(close_ma5_ratio)
                    local_data_dict['range_pos_20'].append(range_pos_20)
                    local_data_dict['body_to_range'].append(body_to_range)
                elif feature_plan_name == 'feature_test':
                    # ema10 = data_0_tmp["ema10"]
                    # ema20 = data_0_tmp["ema20"]
                    # ema60 = data_0_tmp["ema60"]
                    # macd = data_0_tmp["macd"]
                    # signal = data_0_tmp["signal"]
                    # adx = data_0_tmp["adx"]
                    # plus_di = data_0_tmp["plus_di"]
                    # minus_di = data_0_tmp["minus_di"]
                    # atr = data_0_tmp["atr"]
                    # boll_mid = data_0_tmp["boll_mid"]
                    # boll_upper = data_0_tmp["boll_upper"]
                    # boll_lower = data_0_tmp["boll_lower"]
                    # rsi = data_0_tmp["rsi"]
                    # obv = data_0_tmp["obv"]
                    # volume_ma5 = data_0_tmp["volume_ma5"]
                    # close = data_0_tmp["close"]
                    # volume = data_0_tmp["volume"]
                    # ret_5 = data_0_tmp["ret_5"]
                    # hl_range = data_0_tmp["hl_range"]
                    # upper_wick_pct = data_0_tmp["upper_wick_pct"]
                    plus_di = data_0_tmp["plus_di"]
                    minus_di = data_0_tmp["minus_di"]
                    # dist_to_low_20 = data_0_tmp["dist_to_low_20"]
                    # dist_to_high_20 = data_0_tmp["dist_to_high_20"]
                    rsi = data_0_tmp["rsi"]
                    obv = data_0_tmp["obv"]

                    # dist_to_low_20 = data_0_tmp["dist_to_low_20"]
                    volume = data_0_tmp["volume"]
                    close = data_0_tmp["close"]
                    close_ma5_ratio = data_0_tmp["close_ma5_ratio"]
                    range_pos_20 = data_0_tmp["range_pos_20"]
                    body_to_range = data_0_tmp["body_to_range"]

                    check_cols = [
                        # ret_5, hl_range, upper_wick_pct, range_pos_20, dist_to_low_20, dist_to_high_20,
                        plus_di, minus_di, rsi, obv, close_ma5_ratio, range_pos_20, body_to_range,
                        volume, close
                    ]

                    if any(s.isna().any() for s in check_cols):
                        continue
                    cols = [
                        # "ret_5",
                        # "hl_range",
                        # "upper_wick_pct",
                        'plus_di',
                        'minus_di',
                        'rsi',
                        'obv',
                        "volume",
                        "close",
                        "close_ma5_ratio",
                        "range_pos_20",
                        "body_to_range"

                    ]
                    arr = data_0_tmp[cols].to_numpy(dtype=float)
                    if not np.isfinite(arr).all():
                        continue
                    # local_data_dict['ema10'].append(ema10)
                    # local_data_dict['ema20'].append(ema20)
                    # local_data_dict['ema60'].append(ema60)
                    # local_data_dict['macd'].append(macd)
                    # local_data_dict['signal'].append(signal)
                    # local_data_dict['adx'].append(adx)
                    # local_data_dict['plus_di'].append(plus_di)
                    # local_data_dict['minus_di'].append(minus_di)
                    # local_data_dict['atr'].append(atr)
                    # local_data_dict['boll_mid'].append(boll_mid)
                    # local_data_dict['boll_upper'].append(boll_upper)
                    # local_data_dict['boll_lower'].append(boll_lower)
                    # local_data_dict['rsi'].append(rsi)
                    # local_data_dict['obv'].append(obv)
                    # local_data_dict['volume_ma5'].append(volume_ma5)
                    # local_data_dict['volume'].append(volume)
                    # local_data_dict['close'].append(close)

                    # local_data_dict['ret_5'].append(ret_5)
                    # local_data_dict['hl_range'].append(hl_range)
                    # local_data_dict['upper_wick_pct'].append(upper_wick_pct)
                    # local_data_dict['dist_to_low_20'].append(dist_to_low_20)
                    # local_data_dict['dist_to_high_20'].append(dist_to_high_20)

                    local_data_dict['plus_di'].append(plus_di)
                    local_data_dict['minus_di'].append(minus_di)
                    local_data_dict['rsi'].append(rsi)
                    local_data_dict['obv'].append(obv)

                    local_data_dict['volume'].append(volume)
                    local_data_dict['close'].append(close)
                    local_data_dict['close_ma5_ratio'].append(close_ma5_ratio)
                    local_data_dict['range_pos_20'].append(range_pos_20)
                    local_data_dict['body_to_range'].append(body_to_range)
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
    # print(pd.Series(temp_label_list).value_counts())

    return "success"


def get_feature(feature_plan_name):
    # 创建一个字典来存储匹配的结果，这个放置顺序会影响训练准确率
    if feature_plan_name == 'feature_tea_radical_nature':
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
    elif feature_plan_name == 'feature_fuzzy_ma':
        return {
            'plus_di': [],
            'minus_di': [],
            'rsi': [],
            'obv': [],
            'volume': [],
            'close': [],
            'close_ma5_ratio': [],
            'range_pos_20': [],
            'body_to_range': []
        }
    elif feature_plan_name == 'feature_test':
        return {
            'plus_di': [],
            'minus_di': [],
            'rsi': [],
            'obv': [],
            'volume': [],
            'close': [],
            'close_ma5_ratio': [],
            'range_pos_20': [],
            'body_to_range': []
        }
    elif feature_plan_name in FEATURE_PLAN_SPECS:
        return {col: [] for col in FEATURE_PLAN_SPECS[feature_plan_name]}
    else:
        print("未指定feature_plan_name")
        return None


def get_point_to_ts(time_point_step, handle_uneven_samples, strategy_name,
                    feature_plan_name, p2t_name, label_name, temp_data_dict, temp_label_list, assetList,
                    classification, classification_direction):
    if (strategy_name == 'tea_radical_nature'
            or strategy_name == 'fuzzy_ma'
            or strategy_name == 'identify_Market_Types'):
        if p2t_name == "point_to_ts_single":  # 单级别
            res = single_time_level_point_to_ts(assetList, temp_data_dict, temp_label_list, time_point_step,
                                                handle_uneven_samples,
                                                strategy_name, label_name, feature_plan_name,
                                                classification, classification_direction)
        else:
            raise ValueError("p2t_name输入异常")
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
                                                               temp_label_list, assetList, classification,
                                                               classification_direction)
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
                                                               temp_label_list, assetList, None, None)
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
                        fuzzy_ma
                        tea_radical_nature
        feature_plan_name: 不同特征组织方案
                feature_tea_radical_nature
                feature_fuzzy_ma
                feature_test
        p2t_name: point_to_ts_single
        label_name: _label1 _label2 _label5
        name: 标的_级别 ts文件命名
    级别，文件，要进入prepare_dataset后手动设置
    """
    prepare_dataset("_TRAIN", "A_d", 160,
                    50000, True,
                    "fuzzy_ma", "feature_fuzzy_ma",
                    None, "point_to_ts_single", "_label2", 4, "buy")
    prepare_dataset("_TEST", "A_d", 160,
                    10000, True,
                    "fuzzy_ma", "feature_fuzzy_ma",
                    None, "point_to_ts_single", "_label2", 4, "buy")


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
