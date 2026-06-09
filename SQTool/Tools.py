from configparser import ConfigParser
import requests
import json
import pandas as pd
import platform


def read_config(section, item):
    cp = ConfigParser()
    # 这里必须写绝对路径，如果写相对路径，会去找调用这个函数的python文件的相对路径，而不是当前文件的相对路径
    sys_platform = platform.platform().lower()
    if 'windows' in sys_platform:
        path = "D:\\github\\Time-Series-Library-Quant\\Configs\\config.ini"
    else:
        path = "/home/z/data/github/Time-Series-Library-Quant/Configs/config_prd.ini"
    cp.read(path, encoding='utf-8')
    return cp.get(section, item)


def write_config(section, item, value):
    cp = ConfigParser()
    # 这里必须写绝对路径，如果写相对路径，会去找调用这个函数的python文件的相对路径，而不是当前文件的相对路径
    sys_platform = platform.platform().lower()
    if 'windows' in sys_platform:
        path = "D:\\github\\Time-Series-Library-Quant\\Configs\\config.ini"
    else:
        path = "/home/z/data/github/Time-Series-Library-Quant/Configs/config_prd.ini"
    cp.read(path, encoding='utf-8')
    cp.set(section, item, value)
    with open(path, "w") as configfile:
        cp.write(configfile)


def getWorkDay():
    """
    在深交所官网，通过F12，找到交易日历的链接，通过循环传入1~12月，返回当月所有日期，
    其中jybz==‘1’说明是交易日，加入list
    12个月全部查完后，把list转为Dataframe，写入csv
    """
    workday_list = []
    month = 1
    while month < 13:
        res = requests.get("http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month=2026-"+str(month))
        dic = json.loads(res.text)
        for dayDic in dic["data"]:
            if dayDic['jybz'] == '1':
                workday_list.append(dayDic['jyrq'])
        month = month + 1
    df = pd.DataFrame(workday_list, columns=['workday'])
    path = read_config("RMQData_local", "workday_path")
    df.to_csv(path + "workday_list.csv", index=False)


def isWorkDay(filepath, today):
    # 读取工作日列表，遍历看今天是不是交易日
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        if today == row['workday']:
            return True

