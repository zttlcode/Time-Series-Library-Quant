from SQModel import Dataset as SQDataset
from SQRuns import run_quant as run_quant
import SQTool.Tools_time_series_forecast as SQToolTimeSeriesForecast


if __name__ == '__main__':
    """
    功能：TSLQ
    """
    """
    批量组装训练集
    需要测试多个特征时，在BASE_FEATURES中设定基础特征，可以从close开始，然后在USUAL_FEATURES、DISTANCE_CANDIDATES等列表自定义特征，或
    新增列表，新增后修改register_combo_plans函数的入参，将新增列表填入pool，FEATURE_PLAN_SPECS字典就会在基础特征的基础上增量组合pool中
    的特征，combo_sizes必须是元组，表示排列组合方式，combo_sizes=(1, )表示挨个把每1个特征加入基础特征， (1, 3)先组合1个特征、再做33组合。
    最终，FEATURE_PLAN_SPECS字典的key所有特征组名称，值为具体特征值，遍历字典，即可生成对应特征的训练集文件。由于文件名过长会报错，因此我引入
    plan_count标识文件名，按下面先组装训练集，再训练模型，结果在results文件夹下对应标识的result_classification.txt中，其中
    feature_plan_name代表对应特征。目前tea_radical_nature是2分类 fuzzy_ma是4分类。
    2分类关注：
        PR-AUC 提升，且 balanced_accuracy 不降：这个特征值得留
        PR-AUC 不升反降：这个特征大概率删掉
        PR-AUC 持平但 recall 显著提升：看你的策略是否更需要抓信号
        PR-AUC 持平但 precision 变高：看你的策略是否更怕误报
    4分类关注：
        macro-F1 上升，且关键类别 recall 也上升：这个特征大概率有价值；
        macro-F1 下降，但某个关键类别 recall 明显上升：这个特征可能仍然有价值，只是代价在别的类上；
        macro-F1 下降，关键类别也没改善：这个特征大概率没用，甚至有害。
    
    """
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
    #         strategy_name="fuzzy_ma",  # fuzzy_ma tea_radical_nature
    #         feature_plan_name=plan,
    #         plan_count=plan_count,
    #         p2t_name="point_to_ts_single",
    #         label_name="_label2",  # 目前tea_radical_nature是_label2 fuzzy_ma是_label5
    #         classification=4,  # 目前tea_radical_nature是2分类 fuzzy_ma是4分类
    #         classification_direction="buy"
    #     )
    #     SQDataset.prepare_dataset(
    #         flag="_TEST",
    #         name="A_d",
    #         time_point_step=160,
    #         limit_length=10000,
    #         handle_uneven_samples=True,
    #         strategy_name="fuzzy_ma",  # fuzzy_ma tea_radical_nature
    #         feature_plan_name=plan,
    #         plan_count=plan_count,
    #         p2t_name="point_to_ts_single",
    #         label_name="_label2",  # 目前tea_radical_nature是_label2 fuzzy_ma是_label5
    #         classification=4,  # 目前tea_radical_nature是2分类 fuzzy_ma是4分类
    #         classification_direction="buy"
    #     )
    # # 批量运行训练
    # plan_count = 0
    # for plan in plans:
    #     plan_count += 1
    #     print(plan_count, plan)
    #     run_quant.train('A_d',
    #                     160,
    #                     True,
    #                     'fuzzy_ma',  # fuzzy_ma tea_radical_nature
    #                     plan,
    #                     plan_count,
    #                     '_label2',  # 目前tea_radical_nature是_label2 fuzzy_ma是_label5
    #                     'A_d_TimesNet_fuzzy_ma',  # 跟上面对应修改
    #                     'TimesNet',
    #                     4,  # 目前tea_radical_nature是2分类 fuzzy_ma是4分类
    #                     'buy')

    # 训练集组装完成，设置训练或推理参数
    name = 'A_d'
    time_point_step = '160'
    handle_uneven_samples = 'True'
    strategy_name = 'fuzzy_ma'  # fuzzy_ma tea_radical_nature
    feature_plan_name = 'feature_fuzzy_ma'  # feature_all feature_tea_radical_nature feature_fuzzy_ma feature_test
    label_name = '_label2'  # _label2 _label5
    model_name = 'TimesNet'  # ClassCNN  ClassLSTM  Informer Nonstationary_Transformer
    model_id = name + '_' + model_name + '_' + strategy_name  # 区别不同训练系数  a800_60_market  A_15_tea  A_d_pred
    classification = 4
    classification_direction = 'buy'

    # # 单独组装训练集
    # SQDataset.prepare_train_dataset()
    # # 单次运行训练
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
    #                     classification_direction)  # 预测行情 True 是3分类预测行情  False 是4分类预测交易点
    # 实盘运行推理
    run_quant.inference_live(name,
                             time_point_step,
                             strategy_name,
                             feature_plan_name,
                             model_id,
                             model_name,
                             classification,
                             classification_direction)

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
