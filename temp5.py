import torch
import torch.nn as nn
import models.ClassLSTM as ClassLSTM

# 定义 LSTM 模型
net = ClassLSTM.Model(None)

# 生成随机输入数据
x = torch.randn(16, 7, 1, 200)  # (batch_size=16, features=6, channels=1, time_steps=500)
x = x.squeeze(2).permute(0, 2, 1)  # 变成 (batch_size=16, time_steps=500, features=6)

# **遍历每一层，查看形状**
for layer in net.children():
    x = layer(x)
    # 对于 LSTM 层，输出是一个元组 (output, (h_n, c_n))，我们需要取出 output 部分
    if isinstance(layer, nn.LSTM):
        x = x[0]  # 取出 LSTM 层的 output 部分
    print(f"{layer.__class__.__name__} output shape:\t {x.shape}")
