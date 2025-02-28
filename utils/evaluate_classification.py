from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef
)

"""
指标说明：
Accuracy (准确率): 正确预测占总样本的比例
Precision (精确率):
    Macro: 各类别精确率的未加权平均
    Micro: 全局统计TP/FP计算
    Weighted: 按样本量加权的平均
Recall (召回率): 类似精确率的计算方式
F1 Score: 精确率和召回率的调和平均
Per-Class Metrics: 每个类别的独立指标值
Confusion Matrix (混淆矩阵): 分类结果的详细矩阵
    混淆矩阵的行表示真实类别，列表示预测类别
Cohen's Kappa: 考虑随机因素的一致性评估
Matthews Corrcoef: 适用于不平衡数据集的相关系数
    各指标取值范围均为0-1，除了Matthews相关系数(-1到+1)

"""

def evaluate_classification(predictions, trues):
    """评估分类模型的多种指标

    参数:
        predictions (array-like): 模型预测的类别标签
        trues (array-like): 真实的类别标签

    返回:
        dict: 包含各类评估指标的字典
    """
    metrics = {}

    # 基础指标
    metrics["accuracy"] = accuracy_score(trues, predictions)

    # 多分类模式下的宏/微平均
    avg_modes = ["macro", "micro", "weighted"]
    for avg in avg_modes:
        # 当某些类别不存在时，zero_division=0 参数会避免除零错误
        metrics[f"precision_{avg}"] = precision_score(trues, predictions, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(trues, predictions, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(trues, predictions, average=avg, zero_division=0)

    # 各类别单独指标
    metrics["precision_per_class"] = precision_score(trues, predictions, average=None, zero_division=0).tolist()
    metrics["recall_per_class"] = recall_score(trues, predictions, average=None, zero_division=0).tolist()
    metrics["f1_per_class"] = f1_score(trues, predictions, average=None, zero_division=0).tolist()

    # 高级指标
    metrics["confusion_matrix"] = confusion_matrix(trues, predictions).tolist()
    metrics["cohen_kappa"] = cohen_kappa_score(trues, predictions)
    metrics["matthews_corrcoef"] = matthews_corrcoef(trues, predictions)

    return metrics

# 示例数据
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 0, 2]

# 计算评估指标
results = evaluate_classification(y_pred, y_true)

# 打印结果
import pprint

pprint.pprint(results)