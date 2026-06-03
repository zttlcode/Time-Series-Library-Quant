import pandas as pd
import numpy as np
import scipy.signal as signal


# ----------------- 指标计算函数 start -----------------

def calculate_ema(df):
    """
    动态计算EMA（指数移动平均）
    包含EMA20、EMA50及动态周期EMA
    """
    # 计算基础EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()

    # 计算动态周期EMA
    # volatility = df['close'].pct_change().std() * np.sqrt(252)  # 年化波动率
    # dynamic_period = dynamic_ma_period(volatility)
    # df[f'ema{dynamic_period}'] = df['close'].ewm(span=dynamic_period, adjust=False).mean()

    return df


def calculate_macd(df):
    """
    计算MACD指标（指数平滑移动平均线）
    返回DataFrame新增列：macd, signal, histogram
    """
    # 计算长短EMA
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()

    # MACD线与信号线
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']  # MACD柱状图

    return df


def calculate_adx(df, period=14):
    """
    计算ADX指标（平均趋向指数）
    包含+DI、-DI、ADX三列
    """
    # 计算真实波幅（TR）
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算方向运动（+DM和-DM）
    up_move = high - high.shift()
    down_move = low.shift() - low
    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    # 平滑处理
    alpha = 1 / period
    tr_smma = tr.ewm(alpha=alpha, adjust=False).mean()
    pos_dm_smma = pos_dm.ewm(alpha=alpha, adjust=False).mean()
    neg_dm_smma = neg_dm.ewm(alpha=alpha, adjust=False).mean()

    # 计算方向指标（+DI和-DI）
    pos_di = 100 * (pos_dm_smma / tr_smma)
    neg_di = 100 * (neg_dm_smma / tr_smma)

    # 计算趋向指数（DX）和ADX
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    df['adx'] = dx.ewm(span=period, adjust=False).mean()
    df['plus_di'] = pos_di
    df['minus_di'] = neg_di

    return df


def calculate_atr(df, period=14):
    """
    计算ATR指标（平均真实波幅）
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 计算真实波幅
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # 计算ATR（SMMA平滑）
    df['atr'] = tr.ewm(span=period, adjust=False).mean()

    return df


def calculate_bollinger_bands(df, period=20, n_std=2):
    """
    计算布林带指标
    """
    df['boll_mid'] = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    df['boll_upper'] = df['boll_mid'] + n_std * std
    df['boll_lower'] = df['boll_mid'] - n_std * std
    return df


def calculate_rsi(df, period=14):
    """
    计算相对强弱指数（RSI）
    :param df: 包含价格数据的DataFrame
    :param period: RSI计算周期，默认14
    :return: 添加了RSI列的DataFrame

    一般来说，6日、12日和24日的RSI指标分别称为短期、中期和长期指标。和KDJ指标一样，RSI指标也有超买区和超卖区。
    具体而言，当RSI值在50到70之间波动时，表示当前属于强势状态，如继续上升，超过80时，则进入超买区，极可能在短期内转升为跌。
    反之RSI值在20到50之间时，说明当前市场处于相对弱势，如下降到20以下，则进入超卖区，股价可能出现反弹。

    先来讲述一下在实际操作中总结出来的RSI指标的缺陷。
    （1）周期较短（比如6日）的RSI指标比较灵敏，但快速震荡的次数较多，可靠性相对差些，而周期较长（比如24日）的RSI指标可靠性强，
    但灵敏度不够，经常会“滞后”的情况。
    （2）当数值在40到60之间波动时，往往参考价值不大，具体而言，当数值向上突破50临界点时，表示股价已转强，
    反之向下跌破50时则表示转弱。不过在实践过程中，经常会出现RSI跌破50后股价却不下跌，以及突破50后股价不涨。
    """
    # 输入数据校验
    if 'close' not in df.columns:
        raise ValueError("DataFrame必须包含'close'列")
    if len(df) < period:
        raise ValueError(f"数据长度不足，至少需要{period}个数据点")

    # 计算价格变化
    delta = df['close'].diff()

    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

    # 计算相对强弱（RS）
    rs = gain / loss

    # 计算RSI
    df['rsi'] = 100 - (100 / (1 + rs))

    # 处理初始值
    # df['rsi'].iloc[:period] = np.nan  # 前period-1个值为NaN
    df.loc[df.index[:period], 'rsi'] = np.nan

    return df


def calculate_obv(df):
    """
    计算能量潮指标(OBV)
    """
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    return df


def calculate_mfi(df, period=14):
    """计算资金流量指标（MFI）"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # 正向/负向资金流
    positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0)
    negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0)

    # 计算比率
    mfi_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    return df


def calculate_volume_ma(df, periods=5):
    """计算成交量移动平均"""
    df[f'volume_ma{periods}'] = df['volume'].rolling(periods).mean()
    return df


def calculate_kdj(df, n=9, m1=3, m2=3):
    low_list = df['low'].rolling(n).min()
    high_list = df['high'].rolling(n).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['k'] = rsv.ewm(alpha=1 / m1).mean()
    df['d'] = df['k'].ewm(alpha=1 / m2).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df

# ----------------- 指标计算函数 end -----------------


# ----------------- 趋势辅助函数 start -----------------
def get_weekly_ema5_direction(df):
    """获取周线EMA50方向（需输入周线数据）"""
    # 此处需要周线级别数据，假设有weekly_df变量
    weekly_df = df.resample('W').last()  # 将时间序列数据 df 按照每周进行分组。对每个周分组，提取该分组中的最后一行数据。
    weekly_df['ema5'] = weekly_df['close'].ewm(span=5).mean()
    return 1 if weekly_df['ema5'].iloc[-1] > weekly_df['ema5'].iloc[-2] else -1


def get_h4_fib_level(df):
    """获取4小时级别斐波那契关键位（需输入4小时数据）"""
    # 此处需要4小时级别数据，假设有h4_df变量
    h4_df = df.resample('4H').last()
    h4_high = h4_df['high'].iloc[-20:].max()
    h4_low = h4_df['low'].iloc[-20:].min()
    return h4_low + (h4_high - h4_low) * 0.382


def is_valid_uptrend(highs, lows):
    """检查是否形成更高的高点和低点"""
    for i in range(1, len(highs)):
        if not (highs[i] > highs[i - 1] and lows[i] > lows[i - 1]):
            return False
    return True


def is_valid_downtrend(highs, lows):
    """检查是否形成更低的高点和低点"""
    for i in range(1, len(highs)):
        if not (highs[i] < highs[i - 1] and lows[i] < lows[i - 1]):
            return False
    return True


def check_pullback_trend(df, trend_dir, fib_threshold, atr_threshold):
    """验证回调幅度"""
    recent_high = df['high'].iloc[-5:].max()
    recent_low = df['low'].iloc[-5:].min()

    if trend_dir == 1:  # 上涨趋势中的回调
        pullback_size = (recent_high - df['low'].iloc[-1]) / (recent_high - recent_low + 1e-8)    # 防止除零
        return pullback_size < fib_threshold and (recent_high - df['low'].iloc[-1]) < atr_threshold
    else:  # 下跌趋势中的反弹
        rebound_size = (df['high'].iloc[-1] - recent_low) / (recent_high - recent_low + 1e-8)    # 防止除零
        return rebound_size < fib_threshold and (df['high'].iloc[-1] - recent_low) < atr_threshold


def count_consecutive_days(df, trend_dir):
    """计算趋势持续天数"""
    count = 0
    for i in range(1, len(df)):
        if trend_dir == 1:
            if df['close'].iloc[-i] > df['ema20'].iloc[-i]:
                count += 1
            else:
                break
        else:
            if df['close'].iloc[-i] < df['ema20'].iloc[-i]:
                count += 1
            else:
                break
    return count


# ----------------- 趋势辅助函数 end -----------------


# ----------------- 震荡辅助函数 start -----------------

def calculate_slope(series):
    """
    计算时间序列斜率（线性回归）
    """
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    return slope


def count_consecutive_range_days(df):
    """
    统计连续满足震荡条件的天数
    """
    # 此处需要历史标注数据，假设有label列记录每日状态
    if 'label' not in df.columns:
        return 0
    current_idx = df.index[-1]
    consecutive_days = 0
    while df.loc[current_idx, 'label'] == 'range':
        consecutive_days += 1
        current_idx -= pd.Timedelta(days=1)
        if current_idx not in df.index:
            break
    return consecutive_days


def detect_false_breakouts(df, lookback=20):
    """
    检测假突破次数
    """
    false_breakouts = 0
    for i in range(1, lookback + 1):
        # 检查是否突破前高/前低
        if df['high'].iloc[-i] > df['boll_upper'].iloc[-i - 1]:
            # 突破后快速回落
            if df['close'].iloc[-i] < df['boll_upper'].iloc[-i - 1]:
                false_breakouts += 1
        elif df['low'].iloc[-i] < df['boll_lower'].iloc[-i - 1]:
            # 突破后快速反弹
            if df['close'].iloc[-i] > df['boll_lower'].iloc[-i - 1]:
                false_breakouts += 1
    return false_breakouts / lookback  # 返回近期假突破频率


def has_upcoming_events(df):
    """
    检查近期是否有重大事件（需接入外部事件数据）
    """
    # 示例：整合财报日历
    # earnings_dates = pd.read_csv('earnings.csv', parse_dates=['date'])
    # df = df.merge(earnings_dates, how='left', left_index=True, right_on='date')
    # 示例：检查财报发布日期
    if 'earnings_date' in df.columns:
        next_5days = df.index[-1] + pd.Timedelta(days=5)
        return df['earnings_date'].between(df.index[-1], next_5days).any()
    return False


# ----------------- 趋势辅助函数 end -----------------


# ----------------- 反转辅助函数 start -----------------

def check_pattern_break(df):
    """改进版形态突破验证"""
    pattern = df['patterns'].iloc[-1]
    if not pattern:
        return {'confirmed': False}

    # 获取当前价格和颈线值
    current_price = df['close'].iloc[-1]
    neckline = pattern['neckline']['current']

    # 根据形态类型设置突破条件
    if pattern['type'] in ['double_top', 'head_shoulder_top']:
        # 顶部形态：价格需连续3日低于颈线
        break_cond = (df['close'].iloc[-3:] < neckline).all()
    elif pattern['type'] in ['double_bottom', 'head_shoulder_bottom']:
        # 底部形态：价格需连续3日高于颈线
        break_cond = (df['close'].iloc[-3:] > neckline).all()
    else:
        return {'confirmed': False}

    return {
        'confirmed': break_cond,
        'pattern_type': pattern['type'],
        'neckline': neckline
    }


def check_divergence(df, lookback=20):
    """检测量价背离（价格新高但MACD/OBV走低）"""
    # MACD背离
    price_highs = df['high'].rolling(lookback).max()
    macd_highs = df['macd'].rolling(lookback).max()
    macd_divergence = (price_highs.diff() > 0) & (macd_highs.diff() < 0)

    # OBV背离
    obv_highs = df['obv'].rolling(lookback).max()
    obv_divergence = (price_highs.diff() > 0) & (obv_highs.diff() < 0)

    return macd_divergence.iloc[-1] or obv_divergence.iloc[-1]


def check_extreme_sentiment(df):
    """检测市场情绪极端（RSI超买/超卖）"""
    rsi = df['rsi'].iloc[-1]
    # mfi = df['mfi'].iloc[-1]

    # 顶部反转条件
    # if (rsi > 70) and (mfi > 80):
    #     return True
    # # 底部反转条件
    # elif (rsi < 30) and (mfi < 20):
    #     return True

    # 顶部反转条件
    if rsi > 70:
        return True
    # 底部反转条件
    elif rsi < 30:
        return True
    return False


def check_ma_crossover(df):
    """检测均线交叉（趋势反转信号）"""
    ema20 = df['ema20'].iloc[-1]
    ema60 = df['ema60'].iloc[-1]
    # 死亡交叉（顶部反转）
    if df['close'].iloc[-5:].max() > ema20 and ema20 < ema60:
        return True
    # 黄金交叉（底部反转）
    elif df['close'].iloc[-5:].min() < ema20 and ema20 > ema60:
        return True
    return False


def check_stop_loss_risk(df, stop_percent):
    """评估止损空间风险"""
    pattern_high = df['high'].max()
    pattern_low = df['low'].min()
    current_price = df['close'].iloc[-1]

    # 顶部反转止损风险
    if current_price < pattern_high:
        risk = (pattern_high * (1 + stop_percent) - current_price) / current_price
    # 底部反转止损风险
    else:
        risk = (current_price - pattern_low * (1 - stop_percent)) / current_price
    return risk > 0.1  # 止损空间过大时触发风险控制


def find_key_points(prices, mode='peaks',
                    min_height_ratio=0.03, min_distance=5):
    """
    使用科学方法检测关键峰谷点
    :param prices: 价格序列
    :param mode: 'peaks'检测高点，'valleys'检测低点
    :param min_height_ratio: 最小高度比例（相对于价格范围）
    :param min_distance: 峰谷间最小间隔
    :return: 峰/谷索引数组
    """
    # 计算动态高度阈值
    price_range = np.ptp(prices)
    height = price_range * min_height_ratio

    # 检测峰谷
    if mode == 'peaks':
        peaks, _ = signal.find_peaks(prices, height=height, distance=min_distance)
        return peaks
    else:
        valleys, _ = signal.find_peaks(-prices, height=height, distance=min_distance)
        return valleys


def is_double_top(prices):
    """简化版双顶形态检测"""
    """检测双顶形态"""
    peaks = find_key_points(prices, mode='peaks')
    if len(peaks) < 2:
        return False
    # 第二个顶不高于第一个顶的3%
    # return abs(prices[peaks[1]] - prices[peaks[0]]) < prices[peaks[0]] * 0.05
    return prices[peaks[1]] < prices[peaks[0]] * 1.08


def is_double_bottom(prices):
    """检测双底形态"""
    valleys = find_key_points(prices, mode='valleys')
    if len(valleys) < 2:
        return False
    # 第二个底不低于第一个底的3%
    return abs(prices[valleys[1]] - prices[valleys[0]]) < prices[valleys[0]] * 0.05


def is_head_shoulder(prices, is_top=True):
    """
    改进版头肩形态检测（支持顶/底）
    :param prices: 价格序列（高点序列检测顶，低点序列检测底）
    :param is_top: 是否为顶部形态
    :return: 是否形成头肩形态
    """
    # 动态检测关键点
    key_points = find_key_points(prices,
                                       mode='peaks' if is_top else 'valleys')

    if len(key_points) < 3:
        return False

    # 确定头肩位置
    main_idx = np.argmax(prices[key_points]) if is_top else np.argmin(prices[key_points])
    left_points = key_points[key_points < key_points[main_idx]]
    right_points = key_points[key_points > key_points[main_idx]]

    if len(left_points) < 1 or len(right_points) < 1:
        return False

    # 验证形态条件
    left_shoulder = left_points[-1]
    right_shoulder = right_points[0]
    head = key_points[main_idx]

    # 头部必须显著高于/低于肩部
    if is_top:
        valid = (prices[head] > prices[left_shoulder] * 1.03 and
                 prices[head] > prices[right_shoulder] * 1.03)
    else:
        valid = (prices[head] < prices[left_shoulder] * 0.97 and
                 prices[head] < prices[right_shoulder] * 0.97)

    return valid


def calculate_neckline(highs, lows, is_top=True):
    """通用颈线计算（支持顶/底）"""
    if is_top:
        # 头肩顶：左右肩低点连线
        left_idx = 1  # 左肩低点
        right_idx = -2  # 右肩低点
        y_values = lows
    else:
        # 头肩底：左右肩高点连线
        left_idx = 1
        right_idx = -2
        y_values = highs

    x = np.array([left_idx, right_idx])
    y = np.array([y_values[left_idx], y_values[right_idx]])

    # 线性回归计算颈线
    slope, intercept = np.polyfit(x, y, 1)
    return {
        'slope': slope,
        'intercept': intercept,
        'current': slope * (len(lows if is_top else highs) - 1) + intercept  # 当前K线位置
    }


def detect_price_patterns(df):
    """动态检测所有价格形态"""
    # 初始化与df长度相同的patterns列表
    patterns = [None] * len(df)
    highs = df['high'].values
    lows = df['low'].values

    # 检测顶部形态
    high_peaks = find_key_points(highs, mode='peaks')
    for i in range(1, len(high_peaks)):
        window = slice(high_peaks[i - 1], high_peaks[i] + 1)
        if is_double_top(highs[window]):
            patterns[high_peaks[i]] = {
                'type': 'double_top',
                'points': high_peaks[i - 1:i + 1],
                'neckline': calculate_neckline(highs[window], lows[window], is_top=True)
            }
        elif is_head_shoulder(highs[window], is_top=True):
            patterns[high_peaks[i]] = {
                'type': 'head_shoulder_top',
                'points': high_peaks[i - 1:i + 1],
                'neckline': calculate_neckline(highs[window], lows[window], is_top=True)
            }

    # 检测底部形态
    low_valleys = find_key_points(lows, mode='valleys')
    for i in range(1, len(low_valleys)):
        window = slice(low_valleys[i - 1], low_valleys[i] + 1)
        if is_double_bottom(lows[window]):
            patterns[low_valleys[i]] = {
                'type': 'double_bottom',
                'points': low_valleys[i - 1:i + 1],
                'neckline': calculate_neckline(highs[window], lows[window], is_top=False)
            }
        elif is_head_shoulder(lows[window], is_top=False):
            patterns[low_valleys[i]] = {
                'type': 'head_shoulder_bottom',
                'points': low_valleys[i - 1:i + 1],
                'neckline': calculate_neckline(highs[window], lows[window], is_top=False)
            }

    # 将patterns列表转换为Series
    df['patterns'] = pd.Series(patterns, index=df.index)
    return df

# ----------------- 反转辅助函数 end -----------------


# ----------------- 突破辅助函数 start -----------------

def detect_initial_break(df, lookback):
    """检测初步突破信号"""
    # 20日最高最低
    upper_band = df['high'].rolling(lookback).max().shift(1)
    lower_band = df['low'].rolling(lookback).min().shift(1)

    # 当前价格突破
    up_break = df['high'].iloc[-1] > upper_band.iloc[-1]
    down_break = df['low'].iloc[-1] < lower_band.iloc[-1]

    return up_break, down_break


def check_pullback(df, direction):
    """检测回踩动作"""
    if direction == 'up':
        # 回踩支撑（突破后回调不破前高）
        break_level = df['high'].rolling(20).max().iloc[-4]
        return (df['low'].iloc[-3] < break_level) & (df['close'].iloc[-2] > break_level)
    else:
        # 回抽阻力（突破后反弹不破前低）
        break_level = df['low'].rolling(20).min().iloc[-4]
        return (df['high'].iloc[-3] > break_level) & (df['close'].iloc[-2] < break_level)

# ----------------- 突破辅助函数 end -----------------


# ----------------- 入口函数计算函数 start -----------------
def calculate_indicators(df):
    """整合所有指标计算"""
    df = calculate_ema(df)
    df = calculate_macd(df)
    df = calculate_adx(df)
    df = calculate_atr(df)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_obv(df)
    df = calculate_mfi(df)
    df = calculate_volume_ma(df, periods=5)
    df = detect_price_patterns(df)

    return df


def dynamic_ma_period(volatility):
    """动态调整均线周期"""
    if volatility < 0.1:
        return 10
    elif volatility < 0.3:
        return 20
    else:
        return 60


def sigmoid(x, center=0.5, steepness=10):
    """可配置Sigmoid函数"""
    return 1 / (1 + np.exp(-steepness * (x - center)))


def trend_weight(volatility):
    """趋势权重与波动率负相关
    高波动市场降低趋势权重（因假突破增多）
    """
    return np.clip(1.5 - volatility * 10, 0.1, 1.0)


def range_weight(volatility):
    """震荡权重与波动率正相关
    低波动市场提升趋势权重（趋势更稳定）
    """
    return np.clip(volatility * 8, 0.1, 0.8)


def risk_control_check(probs):
    """冲突概率检查"""
    if probs['trend'] > 0.7 and probs['breakout'] > 0.6:
        print("趋势与突破信号冲突，需人工复核")
    if probs['reversal'] > 0.8 and probs['trend'] > 0.5:
        print("趋势与反转信号冲突，需人工复核")

# ----------------- 入口函数计算函数 end -----------------
