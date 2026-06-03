import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch import nn
from torchvision import transforms


def linear_regression():
    # 读取原始数据
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')

    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    # torch.tensor( dataframe类型.to_numpy(dtype=float) )  矩阵的列名会去掉，并转为张量

    # 本例中，价格几十，成交量几千万
    # 原始数据不做任何预处理，每次epoch打印损失时，报nan，可能是特征之间数据之间差距过大，通过特征缩放矫正
    # 且差距大会导致权重不均，收敛慢
    # 特征缩放有两种方式：
    # 1、标准化，将特征缩放到均值为0，标准差为1的分布，没有范围
    # mean = features.mean(dim=0)  # 计算每个特征的均值
    # std = features.std(dim=0)  # 计算每个特征的标准差
    # eps = 1e-8  # 为了避免除以零的情况，我们可以设置一个小的正数作为分母
    # features = (features - mean) / (std + eps)  # 标准化数据：每个特征值减去其均值并除以其标准差

    # 2、最大最小值，范围固定在[0,1]
    min_vals = torch.min(features, dim=0)[0]  # 计算每列特征的最小值和最大值
    max_vals = torch.max(features, dim=0)[0]
    features = (features - min_vals) / (max_vals - min_vals)  # 对每个特征进行最小-最大缩放

    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)  # 标签不做处理

    # 构建随机小批量迭代器
    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    data_iter = load_array((features, labels), batch_size)

    # 内部协变量偏移问题指的是在训练过程中，由于每一层神经网络的参数不断更新，导致每一层输入的分布也会随之发生变化。
    # 这种变化进而会影响下一层的训练，使其变得更加困难，可能需要花费更长的时间来训练，因此要通过批量归一化，
    # 这样，无论网络的前一层输出什么样的分布，BatchNorm都能保证当前层的输入具有合适的尺度
    # 比如输入是2个特征，先用批量归一化分别把两列特征改成均值为0，方差为1，然后2个特征再进入线性层
    # 批量归一化不改变范围，例如批量归一化后，价格变成20左右，成交量变成500万左右
    # 但我只有一层，没有必要用这个
    # net = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
    net = nn.Sequential(nn.Linear(2, 1))
    # Linear在第二位，所以要操作net[1]
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 5
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 通过调用net(X)生成预测并计算损失l（前向传播）
            trainer.zero_grad()
            l.backward()  # 通过进行反向传播来计算梯度  算出每个系数的偏导
            trainer.step()  # 通过调用优化器来更新模型参数  系数 = 系数 - 梯度*学习率  batchsize放在了损失函数里
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1},loss{l:f},p_price{net(features)[-1, 0]:f},true_price{labels[-1, 0]:f}')


# 批量归一化  上面解释了，多层要控制输出才需要用
def linear_regression_batchnorm():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    # 定义均值和标准差
    mean = [0.5, 0.5]  # 特征的均值，假设有两个特征
    std = [0.5, 0.5]  # 特征的标准差，假设有两个特征

    # 定义归一化变换
    normalize = transforms.Normalize(mean=mean, std=std)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        # 创建 Subset，只包含特征
        subset_indices = list(range(len(features)))
        subset = data.Subset(dataset, subset_indices)  # 这就能单独取到features？
        # 对数据集进行归一化处理，仅应用于特征
        subset = subset.transform(transform=normalize)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    # 在这个示例中，我们首先创建了TensorDataset对象，并将特征和标签传递给它。
    # 然后，我们使用Subset来创建一个只包含特征的子集。
    # 最后，我们对这个子集应用归一化变换，这样就只会对特征进行归一化，而不影响标签。

    batch_size = 50
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1), nn.BatchNorm1d(1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 通过调用net(X)生成预测并计算损失l（前向传播）
            trainer.zero_grad()
            l.backward()  # 通过进行反向传播来计算梯度  算出每个系数的偏导
            trainer.step()  # 通过调用优化器来更新模型参数  系数 = 系数 - 梯度*学习率  batchsize放在了损失函数里
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1},loss{l:f}')


# 工具模板
class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 工具模板
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 前面表示不是标量，后面表示某一维不是单个元素
        y_hat = y_hat.argmax(axis=1)  # axis=1 沿着第二个维度（列），找每个样本的最大值的索引
    cmp = y_hat.type(y.dtype) == y  # 索引比较，1说明预测对了
    return float(cmp.type(y.dtype).sum())


# 工具模板
def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # Accumulator里创建2个变量，分别为 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # 每个批量中，正确预测数、预测总数累加起来
    return metric[0] / metric[1]


# 工具模板
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 工具模板
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def softmax_classfi():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()  # 成交量做标准化
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    # 前80%做训练集，后20%做测试集
    data_iter_train = load_array((features[:len(features) * 0.8, :], labels[:len(features) * 0.8, :]), batch_size)
    data_iter_test = load_array((features[len(features) * 0.8:, :], labels[len(features) * 0.8:, :]), batch_size)

    # 初始化参数  每个输出都有一系列系数
    net = nn.Sequential(nn.Linear(2, 2))

    # net = nn.Sequential(nn.Linear(2, 1), nn.BatchNorm1d(1))  图像卷积的中间层才会用
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # y_hat是所有样本，y是所有样本正确类别的概率
    # 所以log的参数是所有样本正确类别的概率列表，然后log对列表里每个值求e为底的对数，还是返回张量
    # 交叉熵返回的就是每个样本正确类别预测概率的负对数值
    # 符号不用担心，对数的x小于1时，y是负数，前面有符号，变成正数，而这里log的参数是概率，他们永远在[0,1]，只是越靠近0，对数反而越大
    # 但熵要越小越好，为什么？
    # 熵值越小，数据越纯，分类越好。也就是log的参数越靠近1越好。也就是y_hat里正确类别的概率越大越好
    # 所以优化就变成了求交叉熵这个函数的最小值。还是通过梯度下降不停更新w和b求它的最小值

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    train_ch3(net, data_iter_train, data_iter_test, loss, num_epochs, trainer)

    # y_hat这2列分别代表 买，卖
    y_hat = torch.tensor([[0.3, 0.7], [0.2, 0.8]])  # 2个样本在2个类别的预测概率
    y = torch.tensor([0, 1])  # y通过马后炮观察得出，这代表从第一个样本开始：不动，买，不动，卖    ！！！！！做分类的人少可能就是标注数据难

    y_hat[[0, 1], y]  # 前面的中括号是要取哪个样本，y代表取每个样本的哪个列
    # 意思就是，y_hat第0个样本，正确答案是第0列，  y_hat第2个样本，正确答案是第0列
    # python里这样对应取值叫做 zip函数，同时遍历两个列表


def mlp_classfi():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()  # 成交量做标准化
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    # 前80%做训练集，后20%做测试集
    data_iter_train = load_array((features[:len(features) * 0.8, :], labels[:len(features) * 0.8, :]), batch_size)
    data_iter_test = load_array((features[len(features) * 0.8:, :], labels[len(features) * 0.8:, :]), batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(2, 32),
                        nn.ReLU(),
                        nn.Linear(32, 2))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);

    batch_size, lr, num_epochs = 32, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_ch3(net, data_iter_train, data_iter_test, loss, num_epochs, trainer)

    """

    # 2024 03 09补充，不论之前是三维，四维，还是更多维，用了Flatten，只能展成2维
    # net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


    这里再捋一遍思路
    线性回归的模型用的是linear
    分类用的是线性回归linear基础上的softmax
    线性回归损失函数是均方误差
    分类用交叉熵
    都是求最小值
    线性回归梯度下降用的sgd
    分类也是sgd

    -----------------------------------------------------------
    # 3维还是Flatten成2维，只是和p4的框架实现相比，这里感知机多加了一层
    # p4里说过，无论多少维，Flatten都是展成为2维
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),  # 第一层输出256，因为有256个单元
                        nn.ReLU(),  # 多的一层用的是relu的激活函数，
                        nn.Linear(256, 10))
    # 与p5手动实现net区别是，p5就一层，数据进去，relu一下，直接输出，所以训练时循环一次net就够了，
    # 这里是两个层，训练时走net函数时，X先展成2维，进nn.Linear做矩阵乘法得到输出，然后输出走relu得到另一个输出，这是第一层，
    # 然后这个relu输出走最后一个nn.Linear做矩阵乘法得到最终10个输出，第二层就是输出层
    """
