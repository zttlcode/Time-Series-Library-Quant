from SQData.Bar import Bar as SQBarEntity
from SQData.Position import PositionEntity as SQPositionEntity
from SQData.Indicator import IndicatorEntity as SQIndicatorEntity


class Asset:
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket):
        # 配置文件
        self.assetsCode = assetsCode
        self.assetsName = assetsName
        self.assetsType = assetsType
        self.assetsMarket = assetsMarket

        # bar数据实例
        self.barEntity = SQBarEntity(assetsCode, timeLevel, isRunMultiLevel, assetsMarket)
        # 指标数据实例
        self.indicatorEntity = SQIndicatorEntity(assetsCode, assetsName, self.barEntity.timeLevel)
        # 订单数据实例
        self.positionEntity = SQPositionEntity(self.indicatorEntity, tradeRule)

    def update_indicatorDF_by_tick(self):
        # 每次tick都在策略中更新指标
        self.indicatorEntity.tick_high = self.barEntity.HighPrice[0]
        self.indicatorEntity.tick_low = self.barEntity.LowPrice[0]
        self.indicatorEntity.tick_close = self.barEntity.ClosePrice[0]
        self.indicatorEntity.tick_time = self.barEntity.DateTimeList[0]
        self.indicatorEntity.tick_volume = self.barEntity.Volume[0]
        # 同时更新bar_dataframe
        rowNum = len(self.barEntity.bar_DataFrame) - 1
        # 数据更新到df的最后一行
        self.barEntity.bar_DataFrame.at[rowNum, 'high'] = self.indicatorEntity.tick_high
        self.barEntity.bar_DataFrame.at[rowNum, 'low'] = self.indicatorEntity.tick_low
        self.barEntity.bar_DataFrame.at[rowNum, 'close'] = self.indicatorEntity.tick_close
        self.barEntity.bar_DataFrame.at[rowNum, 'time'] = self.indicatorEntity.tick_time
        self.barEntity.bar_DataFrame.at[rowNum, 'volume'] = self.indicatorEntity.tick_volume


class Stock(Asset):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket)


class Index(Asset):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket)


class ETF(Asset):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket)


class Crypto(Asset):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule, assetsMarket)
        self.back_test_bar_data = None  # 读取所有的回测数据
        self.back_test_cut_row = None  # 为了从读取的回测bar中，定位到自己要的回测开始时间


def asset_generator(assetsCode, assetsName, timeLevelList, assetsType, tradeRule, assetsMarket):
    """
    :param assetsCode: 给代码
    :param assetsName: 给名字
    :param timeLevelList: 目前只支持5、15、30、60、d这几个级别
    :param assetsType: 资产类型  用于生成不同的资产 stock index ETF crypto
    :param tradeRule: T+1还是T+0
    :param assetsMarket: 所属市场 A HK USA crypto
    :return: 给想要的级别，就能new对应级别的对象，放入列表
    """
    # 判断是否为多级别  多级别在实盘、生成新bar时，会不断用 当日最新成交量-累积成交量，算出当前bar的成交量
    isRunMultiLevel = False
    if len(timeLevelList) > 1:
        isRunMultiLevel = True
    # 根据级别列表，创建各级别资产对象
    assetList = []
    for timeLevel in timeLevelList:
        if assetsType == 'stock':
            assetList.append(Stock(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule,
                                   assetsMarket))
        elif assetsType == 'index':
            assetList.append(Index(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule,
                                   assetsMarket))
        elif assetsType == 'ETF':
            assetList.append(ETF(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule,
                                 assetsMarket))
        elif assetsType == 'crypto':
            assetList.append(Crypto(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType, tradeRule,
                                    assetsMarket))
    return assetList
