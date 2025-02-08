import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Classic LSTM model for time series forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()  # 展平所有时间步
        self.fc1 = nn.Linear(200 * 128 * 2, 256)  # 保持 in_features=128000
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.unsqueeze(2)  # 结果是 (16, 6, 1, 10)
        x_enc = x_enc.squeeze(2).permute(0, 2, 1)  # 变成 (batch_size=16, time_steps=500, features=6)
        output, _ = self.lstm(x_enc)  # 取 output，不取 (h_n, c_n)
        x = self.flatten(output)  # 展平所有时间步
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
