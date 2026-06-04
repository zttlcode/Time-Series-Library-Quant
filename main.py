from SQModel import Dataset as SQDataset
from SQRuns import run_quant as run_quant
import SQTool.Tools_time_series_forecast as SQToolTimeSeriesForecast


if __name__ == '__main__':
    """
    功能：TSLQ
    """
    # 单独组装训练集
    # SQDataset.prepare_train_dataset()
    # 批量组装训练集
    # FEATURE_PLAN_SPECS = SQDataset.FEATURE_PLAN_SPECS
    # plans = list(FEATURE_PLAN_SPECS.keys())
    # plan_count = 0
    # for plan in plans:
    #     plan_count += 1
    #     print(plan_count, plan)
    #     SQDataset.prepare_dataset(
    #         flag="_TRAIN",
    #         name="A_d",
    #         time_point_step=160,
    #         limit_length=50000,
    #         handle_uneven_samples=True,
    #         strategy_name="tea_radical_nature",
    #         feature_plan_name=plan,
    #         plan_count=plan_count,
    #         p2t_name="point_to_ts_single",
    #         label_name="_label5",
    #         classification=2,
    #         classification_direction="buy"
    #     )
    #     SQDataset.prepare_dataset(
    #         flag="_TEST",
    #         name="A_d",
    #         time_point_step=160,
    #         limit_length=10000,
    #         handle_uneven_samples=True,
    #         strategy_name="tea_radical_nature",
    #         feature_plan_name=plan,
    #         plan_count=plan_count,
    #         p2t_name="point_to_ts_single",
    #         label_name="_label5",
    #         classification=2,
    #         classification_direction="buy"
    #     )
    # 训练集组装完成，设置训练或推理参数
    name = 'A_d'
    time_point_step = '160'
    handle_uneven_samples = 'True'
    strategy_name = 'tea_radical_nature'  # fuzzy_nature tea_radical_nature feature_all_v1 feature_basic_plus
    feature_plan_name = 'feature_tea_radical_nature'  # feature_all feature_tea_radical_nature
    label_name = '_label5'  # _label1 _label2
    model_name = 'TimesNet'  # ClassCNN  ClassLSTM  Informer Nonstationary_Transformer
    model_id = name + '_' + model_name + '_' + strategy_name  # 区别不同训练系数  a800_60_market  A_15_tea  A_d_pred
    classification = 2
    classification_direction = 'buy'

    # # 批量运行训练
    # plan_count = 0
    # for plan in plans:
    #     plan_count += 1
    #     print(plan_count, plan)
    #     run_quant.train(name,
    #                     time_point_step,
    #                     handle_uneven_samples,
    #                     strategy_name,
    #                     plan,
    #                     plan_count,
    #                     label_name,
    #                     model_id,
    #                     model_name,
    #                     classification,
    #                     classification_direction)
    # 单次运行训练
    # run_quant.train(name,
    #                 time_point_step,
    #                 handle_uneven_samples,
    #                 strategy_name,
    #                 feature_plan_name,
    #                 None,
    #                 label_name,
    #                 model_id,
    #                 model_name,
    #                 classification,
    #                 classification_direction)
    # 单次运行推理
    # run_quant.inference(name,
    #                     time_point_step,
    #                     handle_uneven_samples,
    #                     strategy_name,
    #                     feature_plan_name,
    #                     label_name,
    #                     model_id,
    #                     model_name,
    #                     classification,
    #                     classification_direction,
    #                     pred_market_type=False)  # 预测行情 True 是3分类预测行情  False 是4分类预测交易点
    # # 实盘运行推理
    # run_quant.inference_live(name,
    #                          time_point_step,
    #                          strategy_name,
    #                          feature_plan_name,
    #                          model_id,
    #                          model_name,
    #                          classification,
    #                          classification_direction)

    """
    功能：时序模型数据处理
    使用步骤：
        需要先用PatchTST训练模型，
        再用preprocess_data_for_OT准备预测数据，并用cut_data_for_OT_pred截取待预测部分，
        然后用PatchTST推理，得到推理结果，
        用compare_actual_pred_close对比预测值与真实值差距
    """
    # SQToolTimeSeriesForecast.preprocess_data_for_OT()  # 将一条股票数据转为时序数据：计算指标并增加OT列。需要提取准备基本的股票历史行情数据
    # SQToolTimeSeriesForecast.cut_data_for_OT_pred()  # 将这条时序数据的末尾部分裁掉，用来给模型推理
    # # 预测时，修改run_longExp.py中is_training、do_predict、data_path
    # SQToolTimeSeriesForecast.compare_actual_pred_close()  # 用matplotlib画出时序模型推理值与真实值的差距
