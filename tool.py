import torch
import torch.nn as nn
import models.ClassLSTM as ClassLSTM
import models.ClassCNN as ClassCNN
import models.CNN_LSTM as CNN_LSTM

import pandas as pd
from exp import exp_classification


def check_voting():
    allStockCode = pd.read_csv("D:/github/RobotMeQ/QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
    df_dataset = allStockCode.iloc[500:]
    n = 1
    for index, row in df_dataset.iterrows():
        asset_code = row['code'][3:]
        # 读取CSV文件
        try:
            df = pd.read_csv("D:/github/Time-Series-Library-Quant/results/" + asset_code + "_prd_result_tpp.csv")
        except FileNotFoundError:
            continue
        # # 过滤出预测正确的行
        # correct_df = df[df["trues"] != df["predictions"]]
        #
        # # 按 predictions_market 和 predictions 分组统计正确的行数
        # correct_counts = correct_df.groupby(["predictions_market", "predictions"]).size().unstack(fill_value=0)
        #
        # # 输出结果
        # print(correct_counts)
        filtered_df = df[df['predictions'].isin([1, 3])]
        #print(filtered_df)
        filtered2_df = filtered_df[filtered_df["predictions"] == filtered_df["predictions_market"]]
        #print(filtered2_df)
        print("old",exp_classification.cal_accuracy(filtered_df["predictions"], filtered_df["trues"]))
        print("new",exp_classification.cal_accuracy(filtered2_df["predictions"], filtered2_df["trues"]))

        n += 1
        if n > 20:
            break


def check_LSTM_shape():
    # 定义 LSTM 模型
    net = ClassLSTM.Model(None)

    # 生成随机输入数据
    x = torch.randn(16, 17, 1, 160)  # (batch_size=16, features=6, channels=1, time_steps=500)
    x = x.squeeze(2).permute(0, 2, 1)  # 变成 (batch_size=16, time_steps=500, features=6)

    # **遍历每一层，查看形状**
    for layer in net.children():
        x = layer(x)
        # 对于 LSTM 层，输出是一个元组 (output, (h_n, c_n))，我们需要取出 output 部分
        if isinstance(layer, nn.LSTM):
            x = x[0]  # 取出 LSTM 层的 output 部分
        print(f"{layer.__class__.__name__} output shape:\t {x.shape}")


def check_CNN_shape():
    """
    我们需要设计一个适合多变量时间序列分类的CNN模型，用于识别不同的买卖点。你的数据包含了6个特征：股票的日线和60分钟线的收盘价与成交量，以及宽基指数的日线收盘价与成交量，总共3000个数值。我们的目标是通过这些特征来进行分类。

    CNN模型架构设计
    输入层：

    输入的数据形状应该是 (batch_size, 6, 500)，即每个样本包含6个特征（每个特征是一个长度为500的时间序列），这与时间序列数据的多变量形式相对应。
    卷积层 (Conv2D)：

    你可以使用一维卷积 (Conv1D) 来处理每个特征序列，或者使用二维卷积 (Conv2D) 来同时处理多个特征。
    由于数据中每个特征是一个时间序列，Conv1D能够在特征维度上处理局部依赖关系，而Conv2D则能捕捉到跨多个特征的局部依赖关系。
    池化层 (MaxPooling2D)：

    池化层用于下采样，减少特征图的维度并帮助模型获得更具泛化能力的特征。使用小的池化窗口，能够有效地减小计算复杂度，并保留重要的时间序列特征。
    Dropout层：

    Dropout层能有效防止模型过拟合，尤其是在训练数据有限的情况下，加入适当的Dropout层能帮助模型进行正则化。
    Flatten层：

    扁平化层用于将卷积层输出的多维特征图转换为一维向量，以便可以输入到全连接层进行分类。
    全连接层 (Dense)：

    最后的全连接层将对抽取的特征进行分类，输出4个类别：1（有效买点）、2（无效买点）、3（有效卖点）、4（无效卖点）。



    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

    # 假设输入形状是 (batch_size, 6, 500, 1) - 6个特征，每个特征500个时间步
    model = Sequential()

    # 第一层卷积：处理输入的6个特征（每个特征是500个时间步）
    model.add(Conv2D(filters=32, kernel_size=(1, 10), activation='relu', input_shape=(6, 500, 1)))
    model.add(Dropout(0.2))  # 防止过拟合

    输入层：input_shape=(6, 500, 1)，表示每个样本有6个特征，每个特征是500个时间步。我们为每个特征添加了一个通道，因此输入的最后一个维度是1。
    卷积层：
        Conv2D(filters=32, kernel_size=(1, 3), activation='relu')：卷积核大小是 (1, 3)，即在时间维度上提取3个连续时间步的特征，
        filters=32 表示第一层输出32个特征图。
    每层后面加上了 Dropout 和 MaxPooling2D 层，帮助降低过拟合，并提取更有用的特征。


    # 第二层卷积：提取更复杂的特征
    model.add(Conv2D(filters=64, kernel_size=(1, 10), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))  # 池化层减少特征图大小
    model.add(Dropout(0.3))


    # 第三层卷积：进一步提取特征
    model.add(Conv2D(filters=128, kernel_size=(1, 10), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))  # 池化层减少特征图大小
    model.add(Dropout(0.3))

    # 扁平化层，将二维卷积层输出展平为一维向量
    model.add(Flatten())

    # 全连接层（Dense），进行分类
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))  # 防止过拟合


    全连接层：在提取完特征后，使用 Dense(128) 层将特征传递给分类层，最后的 Dense(4) 层输出4个类别，对应4种买卖点。
    输出层：使用 softmax 激活函数输出4个类别，分别代表有效买点、无效买点、有效卖点和无效卖点。


    # 输出层，4个类别（有效买点、无效买点、有效卖点、无效卖点）
    model.add(Dense(4, activation='softmax'))


    # 模型总结
    model.summary()
    """
    # 定义 LSTM 模型
    net = ClassCNN.Model(None)

    X = torch.rand(size=(64, 17, 1, 160), dtype=torch.float32)
    # X = torch.rand(size=(64, 17, 160), dtype=torch.float32)

    for layer in net.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)


def draw_CNN():
    # 定义 CNN 模型
    net = ClassCNN.Model(None)

    X = torch.rand(size=(1, 17, 1, 160), dtype=torch.float32)
    # 导出为 ONNX 文件
    torch.onnx.export(net, X, "cnn_model.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)# 导出为 ONNX 文件

def draw_LSTM():
    # 定义 CNN 模型
    net = ClassLSTM.Model(None)

    X = torch.rand(size=(1, 160, 17), dtype=torch.float32)
    # 导出为 ONNX 文件
    torch.onnx.export(net, X, "lstm_model.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)# 导出为 ONNX 文件


if __name__ == '__main__':
    # draw_CNN()
    # draw_LSTM()
    # check_LSTM_shape()
    check_CNN_shape()
    # check_voting()
    # 当前级别
    # 有效买入,上涨\震荡 行情,可信  下跌行情否决    3猜错的最多,否决   结论,上涨行情可信,其余否决
    # 有效卖出,下跌     行情,可行  其他行情否决    3猜错的最少       结论,下跌行情可信

    # d级别
    # 有效买入,
    # 有效卖出,上涨 可行

    # index d级别
    # 有效买入,
    # 有效卖出,上涨 可行