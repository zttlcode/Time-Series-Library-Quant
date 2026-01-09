import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    """
    Classic CNN model for time series forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            # 第一层卷积：处理输入的17个特征（每个特征是500个时间步）
            nn.Conv2d(in_channels=17, out_channels=32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合

            # 第二层卷积：提取更复杂的特征
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 池化层减少特征图大小
            nn.Dropout(0.3),

            # 第三层卷积：进一步提取特征
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 池化层减少特征图大小
            nn.Dropout(0.3),

            # 扁平化层，将二维卷积层输出展平为一维向量
            nn.Flatten(),

            # 全连接层（Dense），进行分类
            nn.Linear(5120, 128),  # 计算卷积和池化后的输出维度：128 * 125
            nn.ReLU(),
            nn.Dropout(0.4),

            # 输出层，4个类别（有效买点、无效买点、有效卖点、无效卖点）
            nn.Linear(128, 4)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        , x_mark_enc, x_dec, x_mark_dec, mask=None
        # 假设原始数据形状是 (16, 10, 6)，表示 batch_size=16，10 个时间步，6 个特征
        x = torch.rand(16, 10, 6)  # (batch_size, time_steps, features)

        # 调整维度顺序，将特征放到第二维，时间步放到第三维
        x = x.permute(0, 2, 1)  # 结果是 (16, 6, 10)

        # 在第二个维度（height）上增加一个维度
        x = x.unsqueeze(2)  # 结果是 (16, 6, 1, 10)

        # 查看结果
        print(x.shape)  # 输出形状 (16, 6, 1, 10)

        画图时，x_enc这两行要注释，入参也只剩x_enc
        """
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.unsqueeze(2)  # 结果是 (16, 6, 1, 10)
        output = self.net(x_enc)
        return output

