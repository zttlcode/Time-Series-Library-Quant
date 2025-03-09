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

        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(17, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出 (64, 250)

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 输出 (128, 125)
        )

        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=128,  # 双向LSTM hidden_size*2
            num_heads=4
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        # 假设原始数据形状是 (16, 10, 6)，表示 batch_size=16，10 个时间步，6 个特征
        x = torch.rand(16, 10, 6)  # (batch_size, time_steps, features)

        # 调整维度顺序，将特征放到第二维，时间步放到第三维
        x = x.permute(0, 2, 1)  # 结果是 (16, 6, 10)

        # 在第二个维度（height）上增加一个维度
        x = x.unsqueeze(2)  # 结果是 (16, 6, 1, 10)

        # 查看结果
        print(x.shape)  # 输出形状 (16, 6, 1, 10)
        """
        x_enc = x_enc.permute(0, 2, 1)

        # CNN处理 (batch, 17, 500) → (batch, 128, 125)
        cnn_out = self.cnn(x_enc)

        # 维度转换 (batch, C, L) → (batch, L, C)
        lstm_input = cnn_out.permute(0, 2, 1)

        # LSTM处理 → (batch, 125, 128)
        lstm_out, _ = self.lstm(lstm_input)

        # 注意力计算（需要调整维度顺序）
        attn_in = lstm_out.permute(1, 0, 2)  # (L, batch, C)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, L, C)

        # 聚合特征
        pooled = torch.mean(attn_out, dim=1)

        return self.classifier(pooled)

