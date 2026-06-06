import SQRuns.run_live as run_live
import SQTool.Tools as SQTool


if __name__ == '__main__':
    """
    功能：每日17:30后，运行实盘日线级别
    核心逻辑：利用run_single_level_no_tick.py的run_back_test_no_tick，运行日线级别的策略，直接打印交易点
    适用策略：tea_radical_nature
            fuzzy_nature
            注意，strategy_name为None，就是行情识别，要去QuantData\market_condition_live中看结果
    标的级别：日线
    
    每天运行：
    日线，ETF，tea，带模型：若买入后触发止损，则启动分钟级实盘
    日线，ETF，fuzzy，带模型
    日线，ETF，fuzzy，系数图  每周一次
    
    注意：
        为了给不同策略实盘分配不同文件夹，而我懒得再给仓位对象加入策略名属性，因此，
        运行策略时要把config.ini的position_currentOrders和position_historyOrders的值手动增加策略名后缀
    """
    strategy_name = "fuzzy_ma"
    if strategy_name == "fuzzy_ma":
        SQTool.write_config("SQData", "position_currentOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_currentOrders_fuzzy_ma\\")
        SQTool.write_config("SQData", "position_historyOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_historyOrders_fuzzy_ma\\")
    elif strategy_name == "tea_radical_nature":
        SQTool.write_config("SQData", "position_currentOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_currentOrders_tea_radical_nature\\")
        SQTool.write_config("SQData", "position_historyOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_historyOrders_tea_radical_nature\\")
    else:
        SQTool.write_config("SQData", "position_currentOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_currentOrders\\")
        SQTool.write_config("SQData", "position_historyOrders",
                            "D:\\github\\RobotMeQ_Dataset\\QuantData\\position_historyOrders\\")

    # 每日策略运行结束，模型预测
    # run_live.run_nature_prepare_dataset(strategy_name)  # 把实盘交易点转为测试集数据，然后运行Time-Series-Library-Quant的inference_live推理
    # run_live.run_nature_prepare_dataset_ssh(strategy_name)  # 执行远程服务器上的实盘csv转ts
    run_live.run_live_get_pred(strategy_name)  # 推理完成后，把分类相同的整理出来发消息，并清空文件

