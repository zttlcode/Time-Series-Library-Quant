import torch
import torch.nn as nn


net = nn.Sequential(
            # 第一层卷积：处理输入的6个特征（每个特征是500个时间步）
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(1, 5), padding=(0, 2)),
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
            nn.Linear(6400, 128),  # 计算卷积和池化后的输出维度：128 * 125
            nn.ReLU(),
            nn.Dropout(0.4),

            # 输出层，4个类别（有效买点、无效买点、有效卖点、无效卖点）
            nn.Linear(128, 4)
        )

X = torch.rand(size=(16, 7, 1, 200), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
