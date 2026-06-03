import json
from SQTool import Tools as SQTools


class PositionEntity:
    def __init__(self, indicatorEntity, tradeRule):
        # 订单数据 二维的字典结构
        self.currentOrders = {}  # 记录买入订单 用于在策略里判断是否有仓位可以卖
        self.historyOrders = {}  # 记录已完成订单，卖出时更新，计算收益
        self.orderNumber = 0  # 每买一单+1
        self.money = 50000000  # 总资产5000万
        self.trade_point_list = []  # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
        self.tradeRule = tradeRule  # T+1是1，T+0是0

        # 尝试读取currentOrders JSON文件
        try:
            with open(SQTools.read_config("SQData", "position_currentOrders")
                      + "position_"
                      + indicatorEntity.IE_assetsCode
                      + "_"
                      + indicatorEntity.IE_timeLevel
                      + ".json", 'r') as file:
                self.currentOrders = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass

        # 尝试读取historyOrders JSON文件
        try:
            with open(SQTools.read_config("SQData", "position_historyOrders")
                      + "position_"
                      + indicatorEntity.IE_assetsCode
                      + "_"
                      + indicatorEntity.IE_timeLevel
                      + ".json", 'r') as file:
                self.historyOrders = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass

    def updateCurrentOrders(self, indicatorEntity):
        with open(SQTools.read_config("SQData", "position_currentOrders")
                  + "position_"
                  + indicatorEntity.IE_assetsCode
                  + "_"
                  + indicatorEntity.IE_timeLevel
                  + ".json", 'w') as file:
            json.dump(self.currentOrders, file)


def buy(positionEntity, indicatorEntity, price, volume):
    positionEntity.orderNumber += 1  # 订单编号更新
    key = "order" + str(positionEntity.orderNumber)  # 准备key
    positionEntity.currentOrders[key] = {'openPrice': price,
                                         'openDateTime': indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                         'volume': volume,
                                         'highPrice': price,
                                         'stopLoss': 0,
                                         'coolDownTime': indicatorEntity.tick_time.strftime('%Y-%m-%d')
                                         }
    # 将仓位信息保存到文件
    with open(SQTools.read_config("SQData", "position_currentOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'w') as file:
        json.dump(positionEntity.currentOrders, file)


def sell(positionEntity, indicatorEntity):
    key = list(positionEntity.currentOrders.keys())[0]  # 把当前仓位的第一个卖掉
    price = indicatorEntity.tick_close
    # 给这个要卖的key，增加关闭价格、交易时间
    positionEntity.currentOrders[key]['closePrice'] = price
    positionEntity.currentOrders[key]['closeDateTime'] = indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
    # 计算本单收益 =（卖价-买入价）* 交易量 - 千分之一印花税 - 买卖两次的手续费万分之3
    positionEntity.currentOrders[key]['pnl'] = ((price - positionEntity.currentOrders[key]['openPrice'])
                                                * positionEntity.currentOrders[key]['volume']
                                                - price * positionEntity.currentOrders[key]['volume'] * 1 / 1000
                                                - (price + positionEntity.currentOrders[key]['openPrice'])
                                                * positionEntity.currentOrders[key]['volume'] * 3 / 10000)
    positionEntity.historyOrders[indicatorEntity.tick_time.strftime('%Y%m%d%H%M')] = positionEntity.currentOrders.pop(key)  # 把卖的订单，从当前仓位列表里，复制到历史仓位列表里

    # 将交易记录保存到文件
    with open(SQTools.read_config("SQData", "position_historyOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'w') as file:
        json.dump(positionEntity.historyOrders, file)

    # 清空仓位信息
    with open(SQTools.read_config("SQData", "position_currentOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'w') as file:
        json.dump({}, file)


def stopLoss(positionEntity, indicatorEntity, strategy_result):
    key = list(positionEntity.currentOrders.keys())[0]  # 当前仓位只有一个
    # 下面两个条件2选1
    if positionEntity.currentOrders[key]['stopLoss'] == 0:  # 还没偷偷止损
        # 按最高价更新止损线
        if indicatorEntity.tick_close > positionEntity.currentOrders[key]['highPrice']:
            # 更新最高价格
            positionEntity.currentOrders[key]['highPrice'] = indicatorEntity.tick_close
            # 更新止损标志位到文件
            positionEntity.updateCurrentOrders(indicatorEntity)
        # 判断是要到止损位
        elif indicatorEntity.tick_close <= positionEntity.currentOrders[key]['highPrice'] * 0.98:
            # 判断目前仓位是不是今天买的
            if positionEntity.currentOrders[key]['coolDownTime'] != indicatorEntity.tick_time.strftime('%Y-%m-%d')\
                    or positionEntity.tradeRule == 0:  # 若以前买的或T+0：止损。否则等明天
                # 现在到止损位了，我偷偷止损，但不告诉策略，策略还以为我现在持有仓位
                positionEntity.currentOrders[key]['coolDownTime'] = indicatorEntity.tick_time.strftime('%Y-%m-%d')
                positionEntity.currentOrders[key]['stopLoss'] = 1

                # 推送消息
                strategy_result.send_msg("stopLoss", indicatorEntity, None, "触发止损位，卖")
                # 更新止损标志位到文件
                positionEntity.updateCurrentOrders(indicatorEntity)

    if positionEntity.currentOrders[key]['stopLoss'] == 1:  # 已经止损了
        # 判断价格涨回来没
        if indicatorEntity.tick_close >= positionEntity.currentOrders[key]['highPrice'] * 0.98:
            # 若相等：今天刚止损卖了，还在CD，不回头
            if positionEntity.currentOrders[key]['coolDownTime'] != indicatorEntity.tick_time.strftime('%Y-%m-%d'):
                # 否则，涨回来了，就当这次止损没发生过，更新锁、置空止损标志位
                positionEntity.currentOrders[key]['coolDownTime'] = indicatorEntity.tick_time.strftime('%Y-%m-%d')
                positionEntity.currentOrders[key]['stopLoss'] = 0

                # 推送消息
                strategy_result.send_msg("stopLoss", indicatorEntity, None, "认错，亏个手续费，重新买回来")
                # 更新仓位文件
                positionEntity.updateCurrentOrders(indicatorEntity)
