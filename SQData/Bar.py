from datetime import timedelta
import pandas as pd
import numpy as np
from SQTool import Tools as SQTools

"""
只适用于A股
    Bar文件里，绝大部分代码都是在把tick转为bar,代码也不多，但很多细节逻辑却很复杂,
    因为我要根据不同的级别生成bar，日线的，60分钟，15分钟等等。这就需要判断级别，判断当前收盘开盘时间，
    bar数据转为模拟的实时数据要改时间，模拟的实时数据又要改回来，还要把实时的数据更新到bar文件里。
    而且，实盘和回测，有些许区别，也要在这里考虑到。
    所以看起来bar与tick转来转去很简单，但实现起来会发现很多细节问题
"""


class Bar:
    def __init__(self, assetsCode, timeLevel, isRunMultiLevel, assetsMarket):
        # 配置文件
        self.timeLevel = timeLevel
        self.bar_num = 250  # 够多少个bar才计算指标，也是计算指标的时间窗口大小  60时，耗时15毫秒，250时，耗时30毫秒
        self.backtest_bar = (SQTools.read_config("SQData", "backtest_bar")
                             + "bar_"
                             + assetsMarket
                             + "_"
                             + assetsCode
                             + "_"
                             + timeLevel
                             + ".csv")  # 几年的bar数据做回测
        self.backtest_tick = (SQTools.read_config("SQData", "backtest_tick")
                              + "tick_"
                              + assetsMarket
                              + "_"
                              + assetsCode
                              + "_"
                              + timeLevel
                              + ".csv")  # bar数据转为tick
        self.live_bar = (SQTools.read_config("SQData", "live_bar")
                         + "live_bar_"
                         + assetsMarket
                         + "_"
                         + assetsCode
                         + "_"
                         + timeLevel
                         + ".csv")  # 实盘bar历史数据，截取250个

        # tick数据
        self.Tick = ()  # 时间+价格+成交量

        # 记录bar数据，列表结构，每次来新tick就更新列表
        self.DateTimeList = []  # 交易时间
        self.OpenPrice = []  # 开bar价
        self.HighPrice = []  # 最高价
        self.LowPrice = []  # 最低价
        self.ClosePrice = []  # 收bar价
        self.Volume = []  # 成交量
        self.bar_DataFrame = None  # 每个tick都会更新bar的dataframe

        # 控制变量
        self._init = False  # 用于控制要不要执行策略
        self.last_bar_start = None  # 上个bar的时间，每5分钟生成bar时用
        self.isLiveRunning = False  # 在第二个bar生成时，记录第一个bar的价格+成交量
        self.live_count = 1  # 用于控制实盘时，保存上一个bar的历史数据
        self.today_volume = 0  # 用于实盘时，记录当日已更新bar的成交量之和，最新的bar减去此值即为当前bar的成交量
        self.isRunMultiLevel = isRunMultiLevel  # 是否多级别同时运行
        self.multiLevel_tempVolume = 0  # 多级别同时回测5分钟tick时，用来临时累加成交量

    def bar_generator(self):
        # 1、判断当前资产标的级别
        time_to_new = self.decide_time_level()
        if time_to_new:
            # 2、tick满足条件，新增bar
            self.bar_handle("new")
            # 3、如果现在是实盘，做一些额外操作
            if self.isLiveRunning:
                self.live_bar_manage()
            # 4、更新指标
            self.update_bar_DataFrame()
        else:
            # 2、tick不满足新增条件，继续维护当前bar
            self.bar_handle("old")
            # 3、如果现在是实盘，做一些额外操作
            if self.isLiveRunning:
                self.Volume[0] = self.Tick[2] - self.today_volume  # 当前bar的成交量 = 今日总成交量 - 今日已记录bar的成交量
            else:
                self.Volume[0] = self.Tick[2]  # 不是实盘，那tick里大部分都是0，只有最后一个tick是成交量，一直更新就行
                # 回测时，如果成交量大于0说明是当前bar的最后一个tick，要把时间更新为结束时间
                # 解释一下：我拿到的bar是9：30~9：35这5分钟的信息，我把他转为tick，当然要从9：30开始，9：35结束，这在Tick.py
                # 的trans_bar_to_ticks函数里转换了。
                # 回测时当前bar最后一个tick的时间是9：34：59，所以bar的结束时间也是9：34：59，但应该改为9：35。这里转换回去
                # 只有最后一个tick的成交量不为0
                if self.Tick[2] != 0:
                    # 最后一个tick的秒数、微秒数置为0
                    # 最后tick的秒数与0秒差多少，秒数减去这个差值，比如9：30：01.700，就得到了9：30：00.700
                    tempCalc = self.Tick[0] - timedelta(seconds=self.Tick[0].second - 0)
                    # 同理减去毫秒值，9：30：00.700 就变成了 9：30：00.000
                    tempTime = tempCalc - timedelta(microseconds=self.Tick[0].microsecond - 0)
                    if self.isRunMultiLevel:
                        # 5分钟级别tick，用在不同级别同时跑时，用此代码。各跑个的，用下面代码
                        # 减好的时间加5分钟，9：30：00.000 变成 9：35：00.000，这就是bar的结束时间，此功能只有回测需要这么做
                        tempLevelTime = tempTime + timedelta(minutes=5)
                        # 把当前分钟提出来，下面做对比用，比如9：35：00.000，提出为35
                        tempMinute = int(tempLevelTime.strftime('%M'))
                        if self.timeLevel == "5":
                            # 最小级别就是5分钟bar，所以bar更新，5分钟不需要特殊操作，结束时间就是9：35：00.000
                            self.DateTimeList[0] = tempLevelTime
                        elif self.timeLevel == "15":
                            # 9：35还不到15分钟级别的9：45，所以把30~35的成交量暂时累加起来
                            self.multiLevel_tempVolume += self.Volume[0]
                            if tempMinute % 15 == 0:
                                # 如果现在是15分钟级别了，比如45，才更新bar结束时间为9：45：00.000
                                self.DateTimeList[0] = tempLevelTime
                                # 把这15分钟累加的三次成交量，一次性赋值给这个bar，再下个bar开启时，把累积值multiLevel_tempVolume置为0
                                self.Volume[0] = self.multiLevel_tempVolume
                        elif self.timeLevel == "30":
                            # 9：35还不到30分钟级别的10：00，所以把30~35的成交量暂时累加起来
                            self.multiLevel_tempVolume += self.Volume[0]
                            if tempMinute % 30 == 0:
                                # 如果现在是30分钟级别了，比如10：00，才更新bar结束时间为10：00：00.000
                                self.DateTimeList[0] = tempLevelTime
                                # 把这30分钟累加的六次成交量，一次性赋值给这个bar，再下个bar开启时，把累积值multiLevel_tempVolume置为0
                                self.Volume[0] = self.multiLevel_tempVolume
                        elif self.timeLevel == "60":
                            # 9：35还不到60分钟级别的10：30，所以把30~35的成交量暂时累加起来
                            self.multiLevel_tempVolume += self.Volume[0]
                            tempHour = int(tempLevelTime.strftime('%H'))
                            if ((tempMinute == 30 and tempHour < 12) or (
                                    tempMinute == 0 and tempHour > 12)):
                                # 如果现在是60分钟级别了，比如上午10：30或下午14：00，才更新bar结束时间
                                self.DateTimeList[0] = tempLevelTime
                                # 把这60分钟累加的12次成交量，一次性赋值给这个bar
                                self.Volume[0] = self.multiLevel_tempVolume
                        elif self.timeLevel == "d":
                            # 9：35还不到日级别的15：00，所以把30~35的成交量暂时累加起来
                            self.multiLevel_tempVolume += self.Volume[0]
                            tempHour = int(tempLevelTime.strftime('%H'))
                            if tempMinute == 0 and tempHour == 15:
                                # 时间到达14：55，如果此时成交量不空，说明是当天最后一个5分钟bar，加5分钟后时间为15：00，此时更新
                                self.DateTimeList[0] = tempLevelTime
                                self.Volume[0] = self.multiLevel_tempVolume
                    else:
                        # 将这个tick的时间转换为相应级别的正确时间
                        if self.timeLevel == "5":
                            self.DateTimeList[0] = tempTime + timedelta(minutes=5)
                        elif self.timeLevel == "15":
                            self.DateTimeList[0] = tempTime + timedelta(minutes=15)
                        elif self.timeLevel == "30":
                            self.DateTimeList[0] = tempTime + timedelta(minutes=30)
                        elif self.timeLevel == "60":
                            self.DateTimeList[0] = tempTime + timedelta(minutes=60)

    def decide_time_level(self):
        # 时间级别不同，对bar的判断也不同
        time_to_new = False
        if self.timeLevel == "5":
            # 当前tick满足5分或0分了；用锁：1、防止头60秒；2、下一次满足5分或0分，证明是新的
            if self.Tick[0].minute % 5 == 0 and self.Tick[0].minute != self.last_bar_start:
                # 更新锁
                self.last_bar_start = self.Tick[0].minute
                # 可以开启新bar
                time_to_new = True
        elif self.timeLevel == "15":
            # 当前tick满足15分或0分了；用锁：1、防止头60秒；2、下一次满足30分或0分，证明是新的
            if self.Tick[0].minute % 15 == 0 and self.Tick[0].minute != self.last_bar_start:
                # 更新锁
                self.last_bar_start = self.Tick[0].minute
                # 可以开启新bar
                time_to_new = True
        elif self.timeLevel == "30":
            """
            当前tick满足30分或0分了；用锁：1、防止头60秒；2、下一次满足30分或0分，证明是新的
            if self.Tick[0].minute % 30 == 0 and self.Tick[0].hour + self.Tick[0].minute != self.last_bar_start:
            当初改成 self.Tick[0].hour + self.Tick[0].minute，是因为有冲突，但现在发现没冲突，忘了当时咋想的，先改回来
            
            时隔2个月，我想起当时咋想的了：
            实盘时，30分钟的bar，我发现10:00和13:30的没有更新，为什么呢？
            
            因为盘前加载历史数据时，bar转tick，最后一个tick是14:30:00.01加到大概14:30:00.28这样的，分钟是30，last_bar_start存30
            然后实盘开启，9:30:00进来，Tick[0].minute是30，last_bar_start也是30，所以time_to_new是False，导致10:00，才变True
            下午也是，上午最后一个tick是11:00:00.01加到大概11:00:00.28这样的，分钟是0，last_bar_start存0
            下午实盘13:00，一看Tick[0].minute是0，last_bar_start也是0，就又没开启新bar
            
            解决办法就是加个小时，或day，这样，昨天最后一个变成30+14，今天的是30+9
            上午最后一个是0+11，下午是0+13，这样就区别开了
            
            5分钟昨天最后一个是55，今天是30，上午最后一个是25，下午是0，不存在此问题
            15分钟昨天最后一个是45，今天是30，上午最后一个是15，下午是0，不存在此问题
            30分钟昨天最后一个是30，今天是30，上午最后一个是0，下午是0
            60分钟昨天最后一个是0，今天是30，上午最后一个是30，下午是0，不存在此问题
            """
            if self.Tick[0].minute % 30 == 0 and self.Tick[0].hour + self.Tick[0].minute != self.last_bar_start:
                # 更新锁
                self.last_bar_start = self.Tick[0].hour + self.Tick[0].minute
                # 可以开启新bar
                time_to_new = True
        elif self.timeLevel == "60":
            # 当前tick上午满足9:30,10:30,11:30，下午满足13，14，15；用锁
            if ((self.Tick[0].minute == 30 and self.Tick[0].hour < 12) or (
                    self.Tick[0].minute == 0 and self.Tick[0].hour > 12)) and \
                    self.Tick[0].hour != self.last_bar_start:
                # 更新锁
                self.last_bar_start = self.Tick[0].hour
                # 可以开启新bar
                time_to_new = True
        elif self.timeLevel == "d":
            # 日线数据tick没有小时，只有日，但多级别用tick有小时，所以只判断日锁就行；用锁
            if self.Tick[0].day != self.last_bar_start:
                # 更新锁
                self.last_bar_start = self.Tick[0].day
                # 可以开启新bar
                time_to_new = True
        return time_to_new

    def bar_handle(self, handle_type):
        if handle_type == "new":
            # 新bar，在列表头插入新增数据
            self.OpenPrice.insert(0, self.Tick[1])
            self.HighPrice.insert(0, self.Tick[1])
            self.LowPrice.insert(0, self.Tick[1])
            self.ClosePrice.insert(0, self.Tick[1])
            self.DateTimeList.insert(0, self.Tick[0])
            self.Volume.insert(0, self.Tick[2])

            # 如果是多级别回测，把多级别回测累计成交量置为0
            if self.isRunMultiLevel and not self.isLiveRunning:
                self.multiLevel_tempVolume = 0
        else:
            # 维护当前bar
            """  tick 合成 bar
            为什么不用tick的最高、最低价合成bar的最高价和最低价呢？
            因为getTick()数据的最高价和最低价是日线的最高和最低，并不是当前tick的最高和最低。
            """
            self.HighPrice[0] = max(self.HighPrice[0], self.Tick[1])
            self.LowPrice[0] = min(self.LowPrice[0], self.Tick[1])
            self.ClosePrice[0] = self.Tick[1]
            self.DateTimeList[0] = self.Tick[0]

    def live_bar_manage(self):
        # 1、只有实盘时，才更新历史数据，新bar追加写入live_bar.csv历史数据文件
        # 实盘当日第二个bar，才开始保存前一个bar的数据。
        if self.live_count >= 2:
            # 刚开盘时加载历史数据，[0]是昨天收盘最后一个bar的数据，已经存过csv了，这里如果不加>=2，就会把昨天最后一个bar再存一遍
            # 实盘只做分钟级，因此DateTimeList是年月日时分秒
            data_list = [self.DateTimeList[0].strftime('%Y-%m-%d %H:%M:00'), self.OpenPrice[1], self.HighPrice[1],
                         self.LowPrice[1],
                         self.ClosePrice[1], self.Volume[1]]
            # print("bar已更新：", data_list)
            # 输入的list为长度6的list（6行rows），而DataFrame需要的是6列(columns)的list。
            # 因此，需要将test_list改为（1*6）的list就可以了。
            data_list = np.array(data_list).reshape(1, 6)
            result = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            result.loc[:, 'time'] = pd.to_datetime(result.loc[:, 'time'])
            # 输出到csv文件
            result.to_csv(self.live_bar, index=False, mode='a', header=False)
            # today_volume是今日已记录bar的成交量，初始为0，更新它。
            # 因为实时tick存的是今日总量，而我想要当前bar的成交量，就要用总量减今日开盘后每个bar累加的成交量
            # 这里是更新每个bar累加，Volume[1]是今天上个bar的量
            self.today_volume += self.Volume[1]

        # 2、如果是下午，且是第一个bar，更新它
        if self.Tick[0].hour > 12 and self.live_count == 1:
            # 因为today_volume初始为0，要把一上午的总量Volume[0]更新给today_volume，方便上面的if以后再减
            self.today_volume += self.Volume[0]

        # 3、进了此函数，说明第一个bar已经生成，今日新bar的计数器+1，以后可以存前一个bar了
        self.live_count += 1

    def update_bar_DataFrame(self):
        # 创建新bar时，相应的指标都要更新
        if len(self.OpenPrice) >= self.bar_num:
            # 判断bar够了
            # 只组装一次DataFrame，以后只拼接
            if len(self.OpenPrice) == self.bar_num:
                # 列表是最新的在0，df逆向存，最新的在最后一个
                comb_bar_list = {'time': self.DateTimeList[::-1], 'open': self.OpenPrice[::-1],
                                 'high': self.HighPrice[::-1], 'low': self.LowPrice[::-1],
                                 'close': self.ClosePrice[::-1], 'volume': self.Volume[::-1]}
                self.bar_DataFrame = pd.DataFrame(comb_bar_list)

            # 只拼接
            # 之前都是根据tick 更新 bar_DataFrame第一行，现在当前bar结束了，bar_DataFrame要新增一行
            if len(self.OpenPrice) > self.bar_num:
                new_comb_row = {'time': [self.DateTimeList[0]], 'open': [self.OpenPrice[0]],
                                'high': [self.HighPrice[0]],
                                'low': [self.LowPrice[0]], 'close': [self.ClosePrice[0]], 'volume': [self.Volume[0]]}
                temp_new_comb_row = pd.DataFrame(new_comb_row)
                # 新增的行排在最后
                self.bar_DataFrame = pd.concat([self.bar_DataFrame, temp_new_comb_row],
                                               ignore_index=True)
            # 指标数据就位，以后可以走策略了
            self._init = True
