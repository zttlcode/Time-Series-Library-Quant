import pandas as pd
import json
from SQTool import Tools as SQTools


# 每个Asset对象 的父类Bar，都有一个指标对象，记录当前Asset对象的各种指标
class IndicatorEntity:
    def __init__(self, assetsCode, assetsName, timeLeve):
        self.tick_high = 0
        self.tick_low = 0
        self.tick_close = 0
        self.tick_time = None
        self.tick_volume = 0
        self.last_cal_time = None  # 上次锁时间，避免一个bar内，每个tick来都执行一遍代码
        self.IE_assetsCode = assetsCode  # 复制资产代码
        self.IE_assetsName = assetsName  # 复制资产名称
        self.IE_timeLevel = timeLeve  # 复制时间级别
        self.signal = {}  # 记录每一次信号  因为激进派策略要记录第几次背离发消息，而收盘后内存清空，因此要持久化

        # 尝试读取signal JSON文件
        try:
            with open(SQTools.read_config("SQData", "indicator_signal")
                      + "signal_"
                      + self.IE_assetsCode
                      + "_"
                      + self.IE_timeLevel
                      + ".json", 'r') as file:
                self.signal = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass

    def updateSignal(self, signalDirection, signalDivergeArea, signalPrice):  # signalDirection 0是底背离  1是顶背离
        isUpdated = False
        if 0 != len(self.signal):  # 说明存过信号
            # 拿当前最新的key
            latest_time_key = None
            for key in self.signal:
                if latest_time_key is None or key > latest_time_key:
                    latest_time_key = key

            if signalDirection == self.signal[latest_time_key]['signalDirection']:  # 方向一致，则比面积
                if signalDivergeArea != self.signal[latest_time_key]['signalDivergeArea']:  # 面积一致，还在一个信号里，不更新，否则
                    self.updateSignalFile(signalDirection, signalDivergeArea, signalPrice)
                    isUpdated = True
            else:  # 方向不一致，清空之前的信号，写入新信号
                self.signal = {}
                self.updateSignalFile(signalDirection, signalDivergeArea, signalPrice)
                isUpdated = True
        else:  # 没存过信号，直接存
            self.updateSignalFile(signalDirection, signalDivergeArea, signalPrice)
            isUpdated = True
        return isUpdated

    def updateSignalFile(self, signalDirection, signalDivergeArea, signalPrice):
        key = str(self.tick_time.strftime('%Y%m%d%H%M'))  # 准备key
        self.signal[key] = {'signalTime': self.tick_time.strftime('%Y-%m-%d %H:%M'),
                            'signalDirection': signalDirection,
                            'signalDivergeArea': signalDivergeArea,
                            'signalPrice': signalPrice
                            }
        # 将信号信息保存到文件
        with open(SQTools.read_config("SQData", "indicator_signal")
                  + "signal_"
                  + self.IE_assetsCode
                  + "_"
                  + self.IE_timeLevel
                  + ".json", 'w') as file:
            json.dump(self.signal, file)


# 一个Asset有多个时间级别，多个级别共用一个多级别指标，来交流指标信息
class IndicatorEntityMultiLevel:
    def __init__(self, assetList):
        self.level_5_diverge = None
        self.level_15_diverge = None
        self.level_30_diverge = None
        self.level_60_diverge = None
        self.level_day_diverge = None
        self.level_day_K = None

        for asset in assetList:
            if asset.barEntity.timeLevel == "5":
                self.level_5_diverge = initDiverge(asset)
            elif asset.barEntity.timeLevel == "15":
                self.level_15_diverge = initDiverge(asset)
            elif asset.barEntity.timeLevel == "30":
                self.level_30_diverge = initDiverge(asset)
            elif asset.barEntity.timeLevel == "60":
                self.level_60_diverge = initDiverge(asset)
            elif asset.barEntity.timeLevel == "d":
                self.level_day_diverge = initDiverge(asset)

    # 每个级别的Asset对象，每次进策略都会更新 多级别指标 的属性值
    def updateDiverge(self, indicatorEntity):
        # EMA DIF DEA MACD MA_5 MA_10 MA_60 K D J RSI6 RSI12 RSI24
        # 暂时只用到这一个指标，以后用到哪个，再加
        if indicatorEntity.IE_timeLevel == "5":
            self.level_5_diverge = getDivergeMsg(indicatorEntity.signal)
        elif indicatorEntity.IE_timeLevel == "15":
            self.level_15_diverge = getDivergeMsg(indicatorEntity.signal)
        elif indicatorEntity.IE_timeLevel == "30":
            self.level_30_diverge = getDivergeMsg(indicatorEntity.signal)
        elif indicatorEntity.IE_timeLevel == "60":
            self.level_60_diverge = getDivergeMsg(indicatorEntity.signal)
        elif indicatorEntity.IE_timeLevel == "d":
            self.level_day_diverge = getDivergeMsg(indicatorEntity.signal)

    def updateDayK(self, windowDF):
        self.level_day_K = windowDF.iloc[-1]['K']


def getDivergeMsg(signal):
    divergeSignal = None
    # 拿当前最新的key
    latest_time_key = None
    for key in signal:
        if latest_time_key is None or key > latest_time_key:
            latest_time_key = key
    signalDirection = signal[latest_time_key]['signalDirection']  # 信号方向
    signalTime = signal[latest_time_key]['signalTime']
    signalPrice = signal[latest_time_key]['signalPrice']
    signalCount = len(signal)  # 信号次数

    if 0 == signalDirection:
        divergeSignal = signalTime + " 第" + str(signalCount) + "次底背离" + str(signalPrice)
    if 1 == signalDirection:
        divergeSignal = signalTime + " 第" + str(signalCount) + "次顶背离" + str(signalPrice)
    return divergeSignal


def initDiverge(asset):
    divergeSignal = None
    # 尝试读取signal JSON文件
    try:
        with open(SQTools.read_config("SQData", "indicator_signal")
                  + "signal_"
                  + asset.assetsCode
                  + "_"
                  + asset.barEntity.timeLevel
                  + ".json", 'r') as file:
            tempSignal = json.load(file)  # 读取各级别当前信号
            if 0 != len(tempSignal):
                divergeSignal = getDivergeMsg(tempSignal)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return divergeSignal


def calEMA(dataFrame, term):
    # 第一个参数是数据，第二个参数是周期
    """计算移动平均值（即EMA）。
    12日EMA1的计算方式是：EMA（12）=前一日EMA（12）× 11/13＋今日收盘价× 2/13
    26日EMA2的计算方式是：EMA（26）=前一日EMA（26）× 25/27＋今日收盘价×2 /27
    """
    # 根据第二个参数term，计算快速（周期是12天）和慢速（周期是26天）的EMA值。
    for i in range(len(dataFrame)):
        if i == 0:  # 如果是第一天，则EMA值用当天的收盘价
            dataFrame.at[i, 'EMA'] = dataFrame.at[i, 'close']
            # 上面是通过df.at的形式访问索引行（比如第i行）和指定标签列（比如EMA列）的数值
            # at方法与之前loc以及iloc方法不同的是，at方法可以通过索引值和标签值访问
            # 而loc以及iloc方法只能通过索引值来访问
        if i > 0:  # 不是第一天，按上面公式计算当天的EMA值。
            dataFrame.at[i, 'EMA'] = ((term - 1) / (term + 1) * dataFrame.at[i - 1, 'EMA']
                                      + 2 / (term + 1) * dataFrame.at[i, 'close'])
    # 计算完成后，把df的EMA列转换成列表类型的对象
    EMAList = list(dataFrame['EMA'])
    return EMAList


def calMACD(dataFrame, shortTerm=12, longTerm=26, DIFTerm=9):
    # 定义计算MACD的方法
    # 得到快速和慢速的EMA值
    shortEMA = calEMA(dataFrame, shortTerm)
    longEMA = calEMA(dataFrame, longTerm)
    # 计算MACD指标中的差离值（即DIF）
    # DIF =今日EMA（12）－今日EMA（26）
    # 注意，shortEMA和longEMA都是列表类型
    # 所以可以通过调用pd.Series方法把它们转换成Series类对象后再直接计算差值。
    dataFrame['DIF'] = pd.Series(shortEMA) - pd.Series(longEMA)
    for i in range(len(dataFrame)):
        if i == 0:  # 第一天
            dataFrame.at[i, 'DEA'] = dataFrame.at[i, 'DIF']  # ix可以通过标签名和索引来获取数据
        if i > 0:
            # 计算差离值的9日EMA（即MACD指标中的DEA）。用差离值计算它的9日EMA，这个值就是差离平均值（DEA）。
            # 今日DEA（MACD）=前一日DEA× 8/10＋今日DIF× 2/10
            dataFrame.at[i, 'DEA'] = ((DIFTerm - 1) / (DIFTerm + 1) * dataFrame.at[i - 1, 'DEA']
                                      + 2 / (DIFTerm + 1) * dataFrame.at[i, 'DIF'])
    # 计算BAR柱状线。
    # BAR=2 × (DIF － DEA)这里乘以2的原因是，在不影响趋势的情况下，从数值上扩大DIF和DEA差值，这样观察效果就更加明显。
    dataFrame['MACD'] = 2 * (dataFrame['DIF'] - dataFrame['DEA'])
    # 返回指定的列
    # return df[['time','DIF','DEA','MACD']]
    # 如果在后面的代码中还要用到df对象的其他列，则如下回df的全部列。
    return dataFrame


def calMACD_area(DataFrame):
    # 定义计算MACD红绿柱面积的方法
    # 拿到macd
    DataFrame = calMACD(DataFrame, 12, 26, 9)
    # 计算好macd面积区域存储在列表里
    macd_area_dic = []

    # 初始化临时变量
    macd_area_sum_temp_red = 0  # 临时累计上涨区域面积
    macd_area_sum_temp_green = 0  # 临时累计下跌区域面积
    dif_extreme_temp_red = -float('inf')  # 红区最大DIF值
    dif_extreme_temp_green = float('inf')  # 绿区最小DIF值

    # 每个区域的价格最值，先初始化
    highest = DataFrame.at[0, 'close']
    lowest = DataFrame.at[0, 'close']

    # 区域变更控制开关
    red_count = 0  # 进入红区累加，变更绿区时，重新归0，开启开关
    green_count = 0  # 进入绿区累加，变更红区时，重新归0，开启开关
    change = False  # 开关，只在变更区域时开启一次，其余时间为关闭状态

    # 遍历整个df
    for index, row in DataFrame.iterrows():
        if row['MACD'] > 0:  # 大于0则用红色
            if green_count > 0:  # 说明刚从绿区进来，开开关，重新归0
                change = True
                green_count = 0
                highest = row['high']  # 最高价初始化为当前区间第一个价格
                dif_extreme_temp_red = row['DIF']  # 初始化红区最大DIF值为当前DIF
            else:  # 说明不是刚进来，保持开关关闭
                change = False
            red_count += 1  # 说明此时是红区状态，一旦下次进去绿区，会被归0
            macd_area_sum_temp_red += row['MACD']  # 红色面积累加
            highest = max(highest, row['high'])  # 记录红区最高价
            dif_extreme_temp_red = max(dif_extreme_temp_red, row['DIF'])  # 记录红区最大DIF值
        elif row['MACD'] < 0:  # 小于0则用绿色
            if red_count > 0:  # 说明刚从红区进来，开开关，重新归0
                change = True
                red_count = 0
                lowest = row['low']  # 最低价初始化为当前区间第一个价格
                dif_extreme_temp_green = row['DIF']  # 初始化绿区最小DIF值为当前DIF
            else:  # 说明不是刚进来，保持开关关闭
                change = False
            green_count += 1  # 说明此时是绿区状态，一旦下次进去红区，会被归0
            macd_area_sum_temp_green += row['MACD']
            lowest = min(lowest, row['low'])
            dif_extreme_temp_green = min(dif_extreme_temp_green, row['DIF'])  # 记录绿区最小DIF值

        # 否则是空文件，都不处理

        if change:
            if red_count == 0:
                # 进入新区域，要把前一个区域的结束时间填上，index代表当前区域，index-1是前一个区域的下标
                macd_area_dic.insert(0, {'area': macd_area_sum_temp_red, 'price': highest,
                                         'dif_extreme': dif_extreme_temp_red,
                                         'time': DataFrame.at[index - 1, 'time']})
                macd_area_sum_temp_red = 0
                highest = 0
                dif_extreme_temp_red = -float('inf')  # 重置红区最大DIF值
            else:
                macd_area_dic.insert(0, {'area': macd_area_sum_temp_green, 'price': lowest,
                                         'dif_extreme': dif_extreme_temp_green,
                                         'time': DataFrame.at[index - 1, 'time']})
                macd_area_sum_temp_green = 0
                lowest = 0
                dif_extreme_temp_green = float('inf')  # 重置绿区最小DIF值

    # 循环结束，判断计算最后一个区域的面积，最后一个就是当前区域
    if red_count > 0:
        # 最后一个是红区
        macd_area_dic.insert(0, {'area': macd_area_sum_temp_red, 'price': highest,
                                 'dif_extreme': dif_extreme_temp_red,
                                 'time': DataFrame.at[len(DataFrame) - 1, 'time']})
    elif green_count > 0:
        # 最后一个是绿区
        macd_area_dic.insert(0, {'area': macd_area_sum_temp_green, 'price': lowest,
                                 'dif_extreme': dif_extreme_temp_green,
                                 'time': DataFrame.at[len(DataFrame) - 1, 'time']})
    # 否则是空文件，都不处理
    result_DataFrame = pd.DataFrame(macd_area_dic)
    # result_DataFrame共2列，面积为正，对应最高价，反之最低价。0是最新数据
    return result_DataFrame, DataFrame


def calMA(dataFrame):
    maIntervalList = [5, 10, 60]
    # 虽然在后文中只用到了5日均线，但这里演示设置3种均线
    # 通过调用rolling方法，还是计算了3日、5日和10日均价，并把计算后的结果记录到当前行的MA_3、MA_5和MA_10这三列中
    # 这样做的目的是为了演示动态创建列的用法。
    for maInterval in maIntervalList:
        dataFrame['MA_' + str(maInterval)] = dataFrame['close'].rolling(window=maInterval).mean()
    return dataFrame


def calKDJ(dataFrame):
    """
    KDJ指标的波动与买卖信号有着紧密的关联，根据KDJ指标的不同取值，可以把这指标划分成三个区域：超买区、超卖区和观望区。
    一般而言，KDJ这三个值在20以下为超卖区，这是买入信号；这三个值在80以上为超买区，是卖出信号；如果这三个值在20到80之间则是在观望区。
    如果再仔细划分一下，当KDJ三个值在50附近波动时，表示多空双方的力量对比相对均衡，当三个值均大于50时，表示多方力量有优势，
    反之当三个值均小于50时，表示空方力量占优势。
    下面根据KDJ的取值以及波动情况，列出交易理论中比较常见的买卖策略。
    （1）KDJ指标中也有金叉和死叉的说法，即在低位K线上穿D线是金叉，是买入信号，反之在高位K线下穿D线则是死叉，是卖出信号。
    （2）一般来说，KDJ指标中的D线由向下趋势转变成向上是买入信号反之，由向上趋势变成向下则为卖出信号。
    （3）K的值进入到90以上为超买区，10以下为超卖区。对D而言，进入80以上为超买区，20以下为超卖区。
    此外，对K线和D线而言，数值50是多空均衡线。如果当前态势是多方市场，50是回档的支持线，即股价回探到KD值是50的状态时，
    可能会有一定的支撑，反之如果是空方市场，50是反弹的压力线，即股价上探到KD是50的状态时，可能会有一定的向下打压的压力。
    （4）一般来说，当J值大于100是卖出信号，如果小于10，则是买入信号。
    当然，上述策略仅针对KDJ指标而言，在现实的交易中，更应当从政策、消息、基本面和资金流等各个方面综合考虑。
    """
    # 计算KDJ
    # 把每一行（即每个交易日）的'MinLow'属性值设置为9天内收盘价（Low）的最小值。
    dataFrame['MinLow'] = dataFrame['low'].rolling(9, min_periods=9).min()
    # 填充NaN数据
    # 如果只执行上面那句，第1到第8个交易日的MinLow属性值将会是NaN，所以要通过下面这行代码，把这些交易日的MinLow属性值设置为
    # 9天内收盘价（Low）的最小值。
    # 如果要把修改后的数据写回到DataFrame中，必须加上inplace=True的参数
    # dataFrame['MinLow'].fillna(value=dataFrame['low'].expanding().min(), inplace=True)
    dataFrame['MinLow'] = dataFrame['MinLow'].fillna(
        dataFrame['low'].expanding().min()
    )
    # 把每个交易日的'MaxHigh'属性值设置为9天内的最高价
    dataFrame['MaxHigh'] = dataFrame['high'].rolling(9, min_periods=9).max()
    # 同样填充前8天的'MaxHigh'属性值
    # dataFrame['MaxHigh'].fillna(value=dataFrame['high'].expanding().max(), inplace=True)
    dataFrame['MaxHigh'] = dataFrame['MaxHigh'].fillna(
        dataFrame['high'].expanding().max()
    )
    # 根据算法计算每个交易日的RSV值
    # df['Close']等变量值是以列为单位，也就是说，在DataFrame中，可以直接以列为单位进行操作
    dataFrame['RSV'] = (dataFrame['close'] - dataFrame['MinLow']) / (dataFrame['MaxHigh'] - dataFrame['MinLow']) * 100
    # 通过for循环依次计算每个交易日的KDJ值
    for i in range(len(dataFrame)):
        if i == 0:  # 第一天
            dataFrame.at[i, 'K'] = 50
            dataFrame.at[i, 'D'] = 50
        if i > 0:
            dataFrame.at[i, 'K'] = dataFrame.at[i - 1, 'K'] * 2 / 3 + 1 / 3 * dataFrame.at[i, 'RSV']
            dataFrame.at[i, 'D'] = dataFrame.at[i - 1, 'D'] * 2 / 3 + 1 / 3 * dataFrame.at[i, 'K']
        dataFrame.at[i, 'J'] = 3 * dataFrame.at[i, 'K'] - 2 * dataFrame.at[i, 'D']
    return dataFrame
