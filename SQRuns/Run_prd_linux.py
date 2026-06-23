import sys

sys.path.append("/home/z/data/github/Time-Series-Library-Quant")

import SQRuns.run_live as run_live
import SQTool.Tools as SQTool
from time import sleep
from datetime import datetime, time, timedelta
from SQRuns import run_quant as run_quant


def run(strategy_name):
    """
    注意：
        为了给不同策略实盘分配不同文件夹，而我懒得再给仓位对象加入策略名属性，因此，
        运行策略时要把config.ini的position_currentOrders和position_historyOrders的值手动增加策略名后缀
    """
    if strategy_name == "fuzzy_ma":
        SQTool.write_config("SQData", "position_currentOrders",
                            "/home/RobotMeQ_Dataset/QuantData/position_currentOrders_fuzzy_ma/")
        SQTool.write_config("SQData", "position_historyOrders",
                            "/home/RobotMeQ_Dataset/QuantData/position_historyOrders_fuzzy_ma/")
        feature_plan_name = 'feature_fuzzy_ma'  # feature_all feature_tea_radical_nature feature_fuzzy_ma feature_test
        classification = 4

    elif strategy_name == "tea_radical_nature":
        SQTool.write_config("SQData", "position_currentOrders",
                            "/home/RobotMeQ_Dataset/QuantData/position_currentOrders_tea_radical_nature/")
        SQTool.write_config("SQData", "position_historyOrders",
                            "/home/RobotMeQ_Dataset/QuantData/position_historyOrders_tea_radical_nature/")
        feature_plan_name = 'feature_tea_radical_nature'
        classification = 2

    name = 'A_d'
    time_point_step = '160'
    model_name = 'TimesNet'  # ClassCNN  ClassLSTM  Informer Nonstationary_Transformer
    model_id = name + '_' + model_name + '_' + strategy_name  # 区别不同训练系数  a800_60_market  A_15_tea  A_d_pred
    classification_direction = 'buy'

    # 每日策略运行结束，模型预测
    run_live.run_nature_prepare_dataset(strategy_name)  # 把实盘交易点转为测试集数据，然后运行Time-Series-Library-Quant的inference_live推理
    print(strategy_name+'数据集准备完成')
    sleep(30)
    # 实盘运行推理
    run_quant.inference_live(name,
                             time_point_step,
                             strategy_name,
                             feature_plan_name,
                             model_id,
                             model_name,
                             classification,
                             classification_direction)
    print(strategy_name+'推理完成')
    sleep(10)
    run_live.run_live_get_pred(strategy_name)  # 推理完成后，把分类相同的整理出来发消息，并清空文件


def seconds_until(target_hour, target_minute):
    now = datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).seconds

def run_prd():
    while True:
        # 只能交易日的0~9:30之间，或交易日15~0之间，手动启
        workday_list = SQTool.read_config("SQT", "QuantData_path") + "workday_list.csv"
        result = SQTool.isWorkDay(workday_list, datetime.now().strftime("%Y-%m-%d"))  # 判断今天是不是交易日  元旦更新
        if result:  # 是交易日
            sleep(seconds_until(16, 0))
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "运行定时任务")
            run("fuzzy_ma")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "完成 position、inference_live、trade_point_live_inference_fuzzy_ma")
            sleep(30)
            run("tea_radical_nature")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "完成 position、inference_live、trade_point_live_inference_tea_radical_nature")
            sleep(seconds_until(7, 0))  # 睡到早上7点，再判断是不是交易日
        else:  # 不是交易日
            sleep(seconds_until(7, 0))  # 直接等24小时，再重新判断


if __name__ == '__main__':
    # run_prd()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "运行任务")
    run("fuzzy_ma")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "完成 position、inference_live、trade_point_live_inference_fuzzy_ma")
    sleep(30)
    run("tea_radical_nature")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "完成 position、inference_live、trade_point_live_inference_tea_radical_nature")