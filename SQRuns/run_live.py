import pandas as pd
import numpy as np
import time
import akshare as ak
import csv
import os
import paramiko
import io
import csv

import glob
from sktime.datasets import write_dataframe_to_tsfile
import shutil
from datetime import datetime


# import sys
# 在cmd窗口python xxx.py 运行脚本时，自己写的from quant找不到quant，必须这样自定义一下python的系统变量，让python能找到
# sys.path.append(r"E:\\PycharmProjects\\robotme")
# 先运行此函数，再导自己的包
import SQData.Asset as SQAsset
from SQData import Identify_market_types_helper as IMTHelper
from SQTool import Message
from SQModel import Dataset as SQDataset
from SQTool import Tools as SQTools


def assemble_ts_data(strategy_name, data, df, assetList):
    # 计算所有指标
    data_0 = SQDataset.build_feature_bank_tea_radical_nature(df)

    # 处理 NaN
    data_0.bfill(inplace=True)
    data_0.ffill(inplace=True)

    # ===== Step 4: 截取最近 160 根 =====
    data_0_tmp = data_0.iloc[-160:].reset_index(drop=True)

    if strategy_name == 'tea_radical_nature':
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
            print("tea_radical_nature指标计算含NA")
            return None

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
            print("tea_radical_nature指标计算含无限值")
            return None

        temp_data_dict = {
            'ret_5': [ret_5],
            'hl_range': [hl_range],
            'upper_wick_pct': [upper_wick_pct],
            'dist_to_low_20': [dist_to_low_20],
            'dist_to_high_20': [dist_to_high_20],
            'range_pos_20': [range_pos_20],
            'volume': [volume],
            'close': [close]
        }
    elif strategy_name == 'fuzzy_ma':
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
            print("fuzzy_ma指标计算含NA")
            return None
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
            print("fuzzy_ma指标计算含无限值")
            return None
        temp_data_dict = {
            'plus_di': [plus_di],
            'minus_di': [minus_di],
            'rsi': [rsi],
            'obv': [obv],
            'volume': [volume],
            'close': [close],
            'close_ma5_ratio': [close_ma5_ratio],
            'range_pos_20': [range_pos_20],
            'body_to_range': [body_to_range]
        }
    else:
        return None

    # ===== Step 5: 标签 =====
    temp_label_list = []
    if data[2] == "buy":
        temp_label_list.append("1")
    else:
        temp_label_list.append("3")

    result_df = pd.DataFrame(temp_data_dict)
    result_series = pd.Series(temp_label_list)

    # ===== Step 6: 写入 ts 文件 =====
    if strategy_name == 'tea_radical_nature':
        feature_plan_name = "feature_tea_radical_nature"
    elif strategy_name == 'fuzzy_ma':
        feature_plan_name = "feature_fuzzy_ma"
    else:
        return None

    problem_name_str = (
            "pred_live_" +
            assetList[0].assetsMarket + "_" +
            assetList[0].assetsCode + "_" +
            assetList[0].barEntity.timeLevel + "_" +
            strategy_name + "_" +
            feature_plan_name + "_160_step"
    )

    if strategy_name == 'tea_radical_nature':
        class_value_list_str = ["1", "2"]
    elif strategy_name == 'fuzzy_ma':
        class_value_list_str = ["1", "2", "3", "4"]
    else:
        return None

    save_path = r"D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts/prediction_live_" + strategy_name

    write_dataframe_to_tsfile(
        data=result_df,
        path=save_path,
        problem_name=problem_name_str,
        class_label=class_value_list_str,
        class_value_list=result_series,
        equal_length=True,
        fold="_TEST"
    )

    write_dataframe_to_tsfile(
        data=result_df,
        path=save_path,
        problem_name=problem_name_str,
        class_label=class_value_list_str,
        class_value_list=result_series,
        equal_length=True,
        fold="_TRAIN"
    )


def run_nature_prepare_dataset(strategy_name):
    # ===== 本地 CSV 目录 =====
    local_live_dir = r"D:\github\RobotMeQ_Dataset\QuantData\live_to_ts"

    # 获取目录下所有 csv 文件
    csv_files = [f for f in os.listdir(local_live_dir) if f.endswith(".csv") and f.startswith(strategy_name)]

    if not csv_files:
        print("没有找到任何 CSV 文件")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(local_live_dir, csv_file)
        print(f"正在处理文件: {csv_path}")

        # ===== Step 1: 读取 CSV =====
        try:
            df = pd.read_csv(csv_path, header=None, dtype={3: str})
        except Exception as e:
            print(f"无法读取 {csv_file}: {e}")
            continue

        if df.empty:
            print(f"文件 {csv_file} 为空，跳过处理")
            continue

        # ===== Step 2: 读取第一行用于元信息 =====
        first_row = df.iloc[0]
        data = first_row.tolist()
        # data 对应关系保持与你原代码一致
        assetList = SQAsset.asset_generator(
            str(data[3]),
            data[4],
            [data[5]],
            data[6],
            int(data[7]),
            data[8]
        )

        # ===== Step 3: 处理行情数据 =====
        try:
            # code = str(data[3])
            # name = data[4]
            # if code.startswith("5") or code.startswith("6"):
            #     code = "sh" + code
            # elif code.startswith("1") or code.startswith("0") or code.startswith("3"):
            #     code = "sz" + code
            # data_0 = ak.stock_zh_a_hist_tx(symbol=code, start_date="20250101", adjust="qfq")
            # if len(data_0) < 250:
            #     print(name, "数据不够250")
            #     continue
            # data_0.rename(
            #     columns={
            #         'date': 'time',
            #         'amount': 'volume'
            #     },
            #     inplace=True
            # )
            csv_path = assetList[-1].barEntity.live_bar
            if not os.path.exists(csv_path):
                print(f"文件不存在，跳过 {assetList[-1].assetsCode}: {csv_path}")
                continue
            # 读取 CSV（有表头）
            data_0 = pd.read_csv(csv_path)
            # 处理数据类型（与 run_daily 保持一致）
            data_0['time'] = pd.to_datetime(data_0['time'])
            # 统一裁剪数据
            data_0 = data_0.reset_index(drop=True)
        except Exception as e:
            print(f"无法解析行情数据 {csv_file}: {e}")
            continue
        # 转化ts文件
        assemble_ts_data(strategy_name, data, data_0, assetList)


def run_nature_prepare_dataset_ssh(strategy_name):
    # 服务器信息
    server_ip = "192.168.2.102"
    username = "root"
    password = "zhao1993"
    docker_container_id = "39fd5ffe1497"
    live_dir = "/home/RobotMeQ_Dataset/QuantData/live_to_ts/"  # Docker 内的 CSV 目录
    local_backup_dir = "D:\\github\\RobotMeQ_Dataset\\QuantData\\live_to_ts\\"  # 本地备份目录

    # 连接服务器
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server_ip, username=username, password=password)

    # 获取 `live` 目录下的所有 CSV 文件
    command_list_csv = (
        f"docker exec {docker_container_id} "
        f"ls {live_dir} | grep -F '{strategy_name}' | grep '\\.csv$'"
    )
    stdin, stdout, stderr = client.exec_command(command_list_csv)
    csv_files = stdout.read().decode().splitlines()

    if not csv_files:
        print("没有找到任何 CSV 文件")
    else:
        for csv_file in csv_files:
            csv_path = f"{live_dir}{csv_file}"
            print(f"正在处理文件: {csv_path}")

            # **Step 1: 临时备份 CSV 文件**
            backup_path = f"/tmp/{csv_file}"  # 先拷贝到服务器宿主机
            command_backup = f"docker cp {docker_container_id}:{csv_path} {backup_path}"
            stdin, stdout, stderr = client.exec_command(command_backup)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                print(f"docker cp 失败: {stderr.read().decode()}")
                continue

            # **Step 2: 下载文件到本地**
            sftp = client.open_sftp()
            local_file_path = os.path.join(local_backup_dir, csv_file)
            sftp.get(backup_path, local_file_path)  # 下载文件到本地
            sftp.close()
            # print(f"已备份到本地: {local_file_path}")

            # 读取 CSV 文件的第一行（存储的是 Python 列表格式的字符串）
            command_read_csv = f"docker exec {docker_container_id} head -n 1 {csv_path}"
            stdin, stdout, stderr = client.exec_command(command_read_csv)
            first_line = stdout.read().decode().strip()

            if not first_line:
                print(f"文件 {csv_file} 为空，跳过处理")
                continue

            data = first_line.split(",")
            # **调用 asset_generator**
            assetList = SQAsset.asset_generator(
                str(data[3]),
                data[4],
                [data[5]],
                data[6],
                int(data[7]),
                data[8]
            )
            # **获取 live_bar 文件路径**
            live_bar = "/home/RobotMeQ_Dataset/QuantData/live/live_bar_"+data[8]+"_"+data[3]+"_"+data[5]+".csv"

            # # **读取 live_bar 文件内容**
            command_read_live_bar = f"docker exec {docker_container_id} cat {live_bar}"
            stdin, stdout, stderr = client.exec_command(command_read_live_bar)
            live_bar_content = stdout.read().decode()

            # **用 Pandas 解析 live_bar 数据**
            try:
                data_0 = pd.read_csv(io.StringIO(live_bar_content), index_col="time", parse_dates=True)
            except Exception as e:
                print(f"无法解析 {csv_path}: {e}")
                continue
            # 转化ts文件
            assemble_ts_data(strategy_name, data, data_0, assetList)

    # **Step 4: 删除 Docker 容器内的所有 CSV 文件**
    command_delete_csv = (
        f"docker exec {docker_container_id} "
        f"sh -c \"rm -f {live_dir}/{strategy_name}*.csv\""
    )
    client.exec_command(command_delete_csv)
    print("所有 Docker 容器中的 CSV 文件已删除")

    # # **Step 5: 删除服务器 `/tmp` 目录下的备份 CSV 文件**
    command_delete_tmp_csv = f"rm -f -- /tmp/{strategy_name}*.csv"
    client.exec_command(command_delete_tmp_csv)
    print("所有服务器 `/tmp` 目录中的备份 CSV 文件已清空")
    # **关闭 SSH 连接**
    client.close()


def run_live_get_pred(strategy_name):
    # ===== 本地 CSV 目录 =====
    local_live_dir = r"D:\github\RobotMeQ_Dataset\QuantData\live_to_ts"

    # 获取目录下所有 csv 文件
    csv_files = [f for f in os.listdir(local_live_dir) if f.endswith(".csv") and f.startswith(strategy_name)]

    if not csv_files:
        print("没有找到任何 CSV 文件")
        return

    backup_rows = []  # 用来备份实盘的所有策略交易点
    temp_result_dict = {}  # 用来整理值得发消息的交易点
    prd_result_files_to_delete = set()  # 在 for 循环前记录要删除的预测结果文件
    position_keys_to_delete = set()     # 候选删除
    position_keys_protected = set()  # 命中过正确预测 → 永久保护
    pred_live_symbols_to_delete = set()  # 清理组装好的测试集文件
    csv_files_to_delete = set()  # 用来保留live_to_ts中还没预测的csv

    for csv_file in csv_files:
        csv_path = os.path.join(local_live_dir, csv_file)
        # print(f"正在处理文件: {csv_path}")

        # ===== Step 1: 读取 CSV =====
        try:
            df = pd.read_csv(csv_path, header=None, dtype={3: str})
        except Exception as e:
            print(f"无法读取 {csv_file}: {e}")
            continue

        if df.empty:
            print(f"文件 {csv_file} 为空，跳过处理")
            continue

        # ===== Step 2: 读取第一行用于元信息 =====
        first_row = df.iloc[0]
        data = first_row.tolist()
        if len(data) >= 9:
            backup_rows.append(data[:9])
        else:
            print(f"{csv_file} 的 data 长度不足 9，跳过备份")

        prob_df_path = SQTools.read_config("SQData", "inference_live")
        df_prd_true_filePath = prob_df_path + data[3] + "_prd_result.csv"
        df_prd_prob_filePath = prob_df_path + data[3] + "_prd_prob.csv"
        if not os.path.exists(df_prd_true_filePath):
            print(data[3] + "预测结果文件不存在")
            continue

        if not os.path.exists(df_prd_prob_filePath):
            print(data[3] + "预测概率文件不存在")
            continue

        # 只有预测文件存在，才认为这个 CSV 被成功消费
        csv_files_to_delete.add(csv_file)

        df_prd_true = pd.read_csv(df_prd_true_filePath)
        df_prd_prob = pd.read_csv(df_prd_prob_filePath)
        # 在 for 循环中，确认文件存在后，记录文件路径
        prd_result_files_to_delete.add(df_prd_true_filePath)
        prd_result_files_to_delete.add(df_prd_prob_filePath)
        # 从仓位管理中删除未达标的仓位
        symbol = data[3]  # 如 sh510300
        timeframe = data[5]  # 如 d
        position_key = f"{symbol}_{timeframe}"

        # ===== 先把当前交易点写入实盘推理结果文件（无论预测对错都写）=====
        # 最大概率类别（1~4）
        max_prob_class_idx = int(df_prd_prob.iloc[0, 0:4].values.argmax()) + 1
        # 最大概率值
        max_prob = round(float(df_prd_prob.iloc[0, 0:4].max()), 3)

        # 组装成：2023-01-16 00:00:00,4.22,sell,3.0,0.899 这种格式
        append_row = [
            str(data[0]),  # 时间
            str(data[1]),  # 交易价格
            str(data[2]),  # 交易行为
            str(max_prob_class_idx),
            str(max_prob)
        ]

        # 输出文件夹：trade_point_live_{strategy_name}_inference
        trade_point_live_dir = os.path.join(
            r"D:\github\RobotMeQ_Dataset\QuantData",
            f"trade_point_live_inference_{strategy_name}"
        )
        os.makedirs(trade_point_live_dir, exist_ok=True)

        # 文件名：A_000100_d.csv 这种
        trade_point_file = os.path.join(
            trade_point_live_dir,
            f"{data[8]}_{data[3]}_{data[5]}.csv"
        )

        # 追加写入，不覆盖，不要列名
        with open(trade_point_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(append_row)

        if df_prd_true['trues'].iloc[0] == df_prd_true['predictions'].iloc[0]:
            # 命中正确预测 → 保护
            position_keys_protected.add(position_key)

            post_msg = (data[4]
                        + "-"
                        + data[3]
                        + "-"
                        + str(data[5])
                        + "：" + data[2] + "："
                        + str(data[1])
                        + " 时间："
                        + data[0]
                        + " 最高概率："
                        + f"{max_prob:.4f}"
                        + "（类"
                        + str(max_prob_class_idx)
                        + "）")
            if data[3] not in temp_result_dict:
                temp_result_dict[data[3]] = []
            temp_result_dict[data[3]].append(post_msg)
        else:
            # 仅当未被保护时，才加入删除候选
            position_keys_to_delete.add(position_key)
        # 统计要删除的训练集文件
        pred_live_symbols_to_delete.add(symbol)

    print("所有交易点验证完成，开始发送消息")
    # 发消息
    if len(temp_result_dict) == 0:
        content_str = "都不符合"
    else:
        content_str = "<br>".join(
            msg for msgs in temp_result_dict.values() for msg in msgs
        )
    mail_msg = Message.build_msg_text_no_entity(strategy_name, content_str)
    mail_list_qq = "mail_list_qq_d"
    res = Message.QQmail(mail_msg, mail_list_qq)
    if res:
        print('消息发送成功')
    else:
        print('消息发送失败')

    for f in csv_files_to_delete:
        file_path = os.path.join(local_live_dir, f)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")

    # 删除本次实际使用过的预测结果文件
    for file_path in prd_result_files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # print(f"已删除预测结果文件: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")

    # 删除实盘仓位中未匹配的交易点，只保留分类正确的交易点
    final_position_keys_to_delete = position_keys_to_delete - position_keys_protected
    position_dir = r"D:\github\RobotMeQ_Dataset\QuantData\position_currentOrders_" + strategy_name

    for key in final_position_keys_to_delete:
        file_path = os.path.join(position_dir, f"position_{key}.json")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # print(f"已删除持仓文件: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")

    pred_live_root_dir = (
        r"D:\github\RobotMeQ_Dataset\QuantData\trade_point_backTest_ts"
        r"\prediction_live_" + strategy_name
    )

    # 删除训练集文件
    for folder_name in os.listdir(pred_live_root_dir):
        folder_path = os.path.join(pred_live_root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for symbol in pred_live_symbols_to_delete:
            # 使用 _symbol_ 作为边界，防止误匹配
            if f"_{symbol}_" in folder_name:
                try:
                    shutil.rmtree(folder_path)
                    # print(f"已删除预测目录: {folder_path}")
                except Exception as e:
                    print(f"删除失败 {folder_path}: {e}")
                break  # 一个目录只需删除一次

    print("数据集组装文件、预测结果文件、不满足推理结果的仓位信息均已删除")


def run_live_run():
    from time import sleep
    from datetime import datetime, time
    from SQTool import Tools as SQTools

    while True:
        # 只能交易日的0~9:30之间，或交易日15~0之间，手动启
        workday_list = SQTools.read_config("SQT", "QuantData_path") + "workday_list.csv"
        result = SQTools.isWorkDay(workday_list, datetime.now().strftime("%Y-%m-%d"))  # 判断今天是不是交易日  元旦更新
        if result:  # 是交易日
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "早上开启进程")

            while (time(9, 30) < datetime.now().time() < time(11, 34)
                   or time(13) < datetime.now().time() < time(15, 4)):
                run_nature_prepare_dataset_ssh("tea_radical_nature")
                run_live_get_pred("tea_radical_nature")
                sleep(360)  # 每过5分钟+60秒，执行一次

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "中午进程停止，等下午")
            sleep(1800)  # 11:30休盘了，等半小时到12:30，开下午盘
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午开启进程")

            while (time(9, 30) < datetime.now().time() < time(11, 34)
                   or time(13) < datetime.now().time() < time(15, 4)):
                run_nature_prepare_dataset_ssh("tea_radical_nature")
                run_live_get_pred("tea_radical_nature")
                sleep(360)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午进程停止，等明天")
            sleep(61200)  # 15点收盘，等17个小时，到第二天8点，重新判断是不是交易日
        else:  # 不是交易日
            sleep(86400)  # 直接等24小时，再重新判断


